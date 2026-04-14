"""
simulate.py  —  Digital Twin City  |  Phase 2 + 3  (routing-fix edition)
=========================================================================
Key fix over previous version
------------------------------
Agents now route on G_live (the per-step working graph that has disaster
edge weights applied) instead of the global G. This means:

  • Flooded / blocked roads are actually avoided by civilians and vehicles.
  • Emergency vehicles take detours around earthquake damage.
  • evac_rate, steps_to_evac, and avg_response_time reflect real conditions.
  • The training CSV has causally consistent columns — disasters cause higher
    travel times, which the LSTM can actually learn from.

Design pattern used
-------------------
Agents no longer call safe_path() at construction time (they can't — G_live
doesn't exist yet). Instead:

  1. Each agent stores only its CURRENT node and a destination node.
  2. On each step(), it receives G_live and calls safe_path(G_live, ...).
  3. If the path is now blocked (empty), it re-routes using G_live.

This means routing is always on the disaster-affected graph.
"""

import json
import os
import random

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
GRAPH_PATH    = "data/city.graphml"
META_PATH     = "data/city_metadata.json"
STEPS         = 24 * 21          # 21 simulated days
NUM_CIVILIANS = 40
NUM_EMERGENCY = 3
RANDOM_SEED   = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load graph + metadata ────────────────────────────────────────────────────
print("Loading graph...")
G_raw = ox.load_graphml(GRAPH_PATH)
G     = ox.convert.to_undirected(G_raw)
largest_cc = max(nx.connected_components(G), key=len)
G     = G.subgraph(largest_cc).copy()
nodes = list(G.nodes)

with open(META_PATH) as f:
    meta = json.load(f)

NUM_ZONES  = meta["num_zones"]
population = {int(k): v for k, v in meta["population"].items()}
infra      = meta["infrastructure"]

zone_nodes: dict = {z: [] for z in range(NUM_ZONES)}
for node, data in G.nodes(data=True):
    zone_nodes[int(data.get("zone", 0))].append(node)

shelter_nodes      = [s["node"] for s in infra.get("shelter",      []) if s["node"] in G.nodes]
hospital_nodes     = [h["node"] for h in infra.get("hospital",     []) if h["node"] in G.nodes]
fire_station_nodes = [f["node"] for f in infra.get("fire_station", []) if f["node"] in G.nodes]

if not hospital_nodes:
    degree = dict(G.degree())
    hospital_nodes = [max(nodes, key=lambda n: degree.get(n, 0))]
if not fire_station_nodes:
    degree = dict(G.degree())
    fire_station_nodes = [sorted(nodes, key=lambda n: degree.get(n, 0))[-2]]

print(f"Shelters: {len(shelter_nodes)}  Hospitals: {len(hospital_nodes)}  "
      f"Fire stations: {len(fire_station_nodes)}")


# ── Routing helper ────────────────────────────────────────────────────────────
def safe_path(graph, source, target, weight="length"):
    """Return shortest path on GRAPH (whatever graph is passed in)."""
    try:
        return nx.shortest_path(graph, source, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


# ── DisasterEvent ─────────────────────────────────────────────────────────────
class DisasterEvent:
    TYPES = ["none", "flood", "fire", "earthquake"]

    def __init__(self, event_type, affected_zone, severity=1.0):
        assert event_type in self.TYPES
        self.event_type    = event_type
        self.affected_zone = affected_zone
        self.severity      = severity
        self.active        = True
        self.step_count    = 0

    def apply(self, G_live, zone_nodes):
        """
        Mutate G_live in-place: inflate edge weights or remove edges.
        Returns the set of blocked (u, v) pairs for the blocked_ratio KPI.

        WHY mutate G_live and not G:
          G is the permanent base graph. G_live is a fresh copy made at the
          start of every timestep. Mutating G_live means next step starts
          clean — disasters don't permanently corrupt the base graph, which
          would make the simulation irreversible and un-reproducible.
        """
        blocked = set()
        z_nodes = set(zone_nodes.get(self.affected_zone, []))

        if self.event_type == "flood":
            for u, v, k, data in list(G_live.edges(keys=True, data=True)):
                if u in z_nodes and v in z_nodes:
                    if random.random() < self.severity * 0.4:
                        # Fully block: remove the edge from G_live so
                        # shortest_path() will route around it.
                        G_live.remove_edge(u, v, k)
                        blocked.add((u, v))
                    else:
                        G_live[u][v][k]["length"] = (
                            data.get("length", 100) * (1 + self.severity * 3)
                        )

        elif self.event_type == "fire":
            hazard = min(self.severity + self.step_count * 0.02, 1.0)
            for u, v, k in list(G_live.edges(keys=True)):
                if u in z_nodes or v in z_nodes:
                    if random.random() < hazard * 0.3:
                        G_live.remove_edge(u, v, k)
                        blocked.add((u, v))

        elif self.event_type == "earthquake":
            for u, v, k in list(G_live.edges(keys=True)):
                if random.random() < self.severity * 0.08:
                    G_live.remove_edge(u, v, k)
                    blocked.add((u, v))

        self.step_count += 1
        return blocked

    def zone_hazard(self):
        if self.event_type == "none":
            return 0.0
        if self.event_type == "fire":
            return min(self.severity + self.step_count * 0.02, 1.0)
        return float(self.severity)


# ── Civilian ──────────────────────────────────────────────────────────────────
class Civilian:
    """
    Routing fix:
      - __init__ stores only current node + destination, NOT a pre-computed path.
      - step(t, G_live, disaster) computes / follows the path on G_live.
      - If the next node on the current path has been disconnected (blocked edge),
        the agent re-routes on G_live from its current position.
    """

    def __init__(self, cid, zone):
        self.cid           = cid
        self.zone          = zone
        self.evacuating    = False
        self.evacuated     = False
        self.steps_to_evac = None

        z_nodes = zone_nodes.get(zone, nodes)
        self.current = random.choice(z_nodes if z_nodes else nodes)
        self.dest    = None
        self.path    = []
        self.idx     = 0

    def _repath(self, graph):
        """Pick a random destination and compute path on GRAPH."""
        candidates = [n for n in nodes if n != self.current]
        self.dest = random.choice(candidates)
        self.path = safe_path(graph, self.current, self.dest)
        self.idx  = 0

    def _path_still_valid(self, graph):
        """
        Check that the NEXT edge on the current path still exists in GRAPH.
        Called before advancing so agents don't walk through blocked roads.
        """
        if not self.path or self.idx >= len(self.path) - 1:
            return False
        u = self.path[self.idx]
        v = self.path[self.idx + 1]
        return graph.has_edge(u, v)

    def start_evacuation(self, graph):
        """Route to nearest reachable shelter on GRAPH (the live graph)."""
        if self.evacuated:
            return
        best_path, best_len = [], float("inf")
        for s in shelter_nodes:
            p = safe_path(graph, self.current, s)   # uses G_live
            if 2 <= len(p) < best_len:
                best_path, best_len = p, len(p)
        if best_path:
            self.path       = best_path
            self.idx        = 0
            self.evacuating = True

    def step(self, t, G_live, disaster=None):
        # Trigger evacuation when disaster hits home zone
        if (disaster and disaster.active
                and disaster.affected_zone == self.zone
                and not self.evacuating
                and not self.evacuated):
            self.start_evacuation(G_live)

        # Re-route if current path is blocked by disaster
        if self.evacuating and not self.evacuated:
            if not self._path_still_valid(G_live):
                self.start_evacuation(G_live)   # re-route on live graph
        else:
            if not self._path_still_valid(G_live):
                self._repath(G_live)

        # Advance one step along path
        if not self.path or self.idx >= len(self.path) - 1:
            if self.evacuating and not self.evacuated:
                self.evacuated     = True
                self.steps_to_evac = t
                return
            self._repath(G_live)
            return

        self.idx    += 1
        self.current = self.path[self.idx]

        if self.evacuating and self.current in shelter_nodes:
            self.evacuated     = True
            self.steps_to_evac = t


# ── EmergencyVehicle ──────────────────────────────────────────────────────────
class EmergencyVehicle:
    """Same routing fix: deploy() and step() receive G_live."""

    def __init__(self, eid):
        self.eid           = eid
        self.current       = random.choice(fire_station_nodes)
        self.path          = []
        self.idx           = 0
        self.deployed      = False
        self.arrived       = False
        self.response_time = None

    def _path_still_valid(self, graph):
        if not self.path or self.idx >= len(self.path) - 1:
            return False
        u = self.path[self.idx]
        v = self.path[self.idx + 1]
        return graph.has_edge(u, v)

    def deploy(self, target_zone, graph):
        if self.deployed:
            return
        z_nodes = zone_nodes.get(target_zone, nodes)
        if not z_nodes:
            return
        target = random.choice(z_nodes)
        path   = safe_path(graph, self.current, target)  # uses G_live
        if len(path) >= 2:
            self.path     = path
            self.idx      = 0
            self.deployed = True

    def step(self, t, G_live, disaster=None):
        if disaster and disaster.active and not self.deployed:
            self.deploy(disaster.affected_zone, G_live)

        # Re-route if path is now blocked
        if self.deployed and not self.arrived:
            if not self._path_still_valid(G_live):
                if disaster:
                    self.deployed = False           # re-trigger deploy
                    self.deploy(disaster.affected_zone, G_live)

        if not self.path or self.idx >= len(self.path) - 1:
            if self.deployed and not self.arrived:
                self.arrived       = True
                self.response_time = t
            return

        self.idx    += 1
        self.current = self.path[self.idx]

        z_nodes = zone_nodes.get(disaster.affected_zone if disaster else -1, [])
        if self.current in z_nodes and not self.arrived:
            self.arrived       = True
            self.response_time = t


# ── Background Vehicle ────────────────────────────────────────────────────────
class Vehicle:
    """Background traffic — also reroutes on G_live if path blocked."""

    def __init__(self, vid):
        self.vid     = vid
        self.current = random.choice(nodes)
        self.dest    = None
        self.path    = []
        self.idx     = 0

    def _repath(self, graph):
        candidates = [n for n in nodes if n != self.current]
        self.dest = random.choice(candidates)
        self.path = safe_path(graph, self.current, self.dest)
        self.idx  = 0

    def _path_still_valid(self, graph):
        if not self.path or self.idx >= len(self.path) - 1:
            return False
        u = self.path[self.idx]
        v = self.path[self.idx + 1]
        return graph.has_edge(u, v)

    def step(self, G_live):
        if not self._path_still_valid(G_live):
            self._repath(G_live)
        if not self.path or self.idx >= len(self.path) - 1:
            return
        self.idx    += 1
        self.current = self.path[self.idx]

    @property
    def current_edge(self):
        if self.idx == 0 or not self.path:
            return None
        return (self.path[self.idx - 1], self.path[self.idx])


# ── Spawn agents ──────────────────────────────────────────────────────────────
print("Spawning agents...")

total_pop = sum(population.values())
civilians = []
cid = 0
for z in range(NUM_ZONES):
    count = max(1, int(NUM_CIVILIANS * population[z] / total_pop))
    for _ in range(count):
        civilians.append(Civilian(cid, z))
        cid += 1

emergency_vehicles = [EmergencyVehicle(i) for i in range(NUM_EMERGENCY)]
vehicles           = [Vehicle(i) for i in range(20)]

print(f"Civilians: {len(civilians)}  Emergency: {len(emergency_vehicles)}  "
      f"Background: {len(vehicles)}")

# ── Disaster schedule ─────────────────────────────────────────────────────────
disaster_schedule = [
    (48,  DisasterEvent("flood",      affected_zone=2, severity=0.7)),
    (120, DisasterEvent("fire",       affected_zone=4, severity=0.6)),
    (240, DisasterEvent("earthquake", affected_zone=0, severity=0.5)),
    (360, DisasterEvent("flood",      affected_zone=5, severity=0.9)),
]
active_disaster = None

# ── Simulation loop ───────────────────────────────────────────────────────────
print("Running simulation...")
logs = []

for t in range(STEPS):

    # Activate scheduled disaster
    for trigger_t, event in disaster_schedule:
        if t == trigger_t:
            active_disaster = event
            print(f"  [t={t:>4}] Disaster: {event.event_type} "
                  f"zone {event.affected_zone} sev={event.severity}")

    if active_disaster and active_disaster.step_count >= 48:
        active_disaster.active = False

    # ── Build G_live for this timestep ───────────────────────────────────────
    # G_live is built ONCE per step and passed to every agent.
    # All routing calls use the same disaster-affected graph.
    # G (base graph) is never modified — fresh copy each step.
    G_live = G.copy()
    blocked_edges = set()
    if active_disaster and active_disaster.active:
        blocked_edges = active_disaster.apply(G_live, zone_nodes)

    blocked_ratio = len(blocked_edges) / max(len(G.edges), 1)

    # Step agents — all receive G_live
    for v in vehicles:
        v.step(G_live)
    for c in civilians:
        c.step(t, G_live, active_disaster)
    for e in emergency_vehicles:
        e.step(t, G_live, active_disaster)

    # ── Metrics ──────────────────────────────────────────────────────────────
    node_counts = {}
    for v in vehicles:
        node_counts[v.current] = node_counts.get(v.current, 0) + 1
    for c in civilians:
        node_counts[c.current] = node_counts.get(c.current, 0) + 1

    vals           = list(node_counts.values())
    active_nodes   = len(node_counts)
    mean_node_load = float(np.mean(vals)) if vals else 0.0
    max_node_load  = float(np.max(vals))  if vals else 0.0

    edge_counts = {}
    for v in vehicles:
        e = v.current_edge
        if e:
            edge_counts[e] = edge_counts.get(e, 0) + 1
    evals          = list(edge_counts.values())
    mean_edge_load = float(np.mean(evals)) if evals else 0.0
    max_edge_load  = float(np.max(evals))  if evals else 0.0

    evacuated_count  = sum(1 for c in civilians if c.evacuated)
    evacuating_count = sum(1 for c in civilians if c.evacuating and not c.evacuated)
    evac_rate        = evacuated_count / max(len(civilians), 1)
    shelter_util     = (sum(1 for c in civilians if c.current in shelter_nodes)
                        / max(len(shelter_nodes), 1))

    arrived = [e.response_time for e in emergency_vehicles if e.response_time is not None]
    avg_response_time = float(np.mean(arrived)) if arrived else -1.0

    zone_hazard = (active_disaster.zone_hazard()
                   if (active_disaster and active_disaster.active) else 0.0)

    # Time / weather
    hour        = t % 24
    day         = t // 24
    day_of_week = day % 7
    is_weekend  = int(day_of_week >= 5)

    morning_peak = np.exp(-((hour - 8.5)  ** 2) / (2 * 1.8 ** 2))
    evening_peak = np.exp(-((hour - 18.0) ** 2) / (2 * 2.2 ** 2))
    midday_bump  = 0.35 * np.exp(-((hour - 13.0) ** 2) / (2 * 2.5 ** 2))
    rush_factor  = 1.0 + 0.9 * morning_peak + 1.1 * evening_peak + midday_bump
    weekend_factor = 0.72 if is_weekend else 1.0

    temperature = 22 + 6 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.8)
    wind = max(0.0, 3 + 1.5 * np.cos(2 * np.pi * hour / 24) + np.random.normal(0, 0.3))

    base = (0.45 * active_nodes + 5.8 * mean_node_load + 3.2 * max_node_load
            + 2.0 * mean_edge_load + 1.5 * max_edge_load)
    disaster_spike = 1.0 + (2.5 * zone_hazard
                            if (active_disaster and active_disaster.active) else 0.0)
    traffic = max(0.0, float(
        base * rush_factor * weekend_factor * disaster_spike + np.random.normal(0, 1.0)
    ))
    aqi = max(0.0, float(
        38 + 0.95 * traffic + 0.55 * temperature - 1.60 * wind
        + 15 * zone_hazard + np.random.normal(0, 2.0)
    ))

    logs.append({
        "timestep":          t,
        "day":               day,
        "day_of_week":       day_of_week,
        "is_weekend":        is_weekend,
        "hour":              hour,
        "num_vehicles":      len(vehicles),
        "active_nodes":      active_nodes,
        "mean_node_load":    mean_node_load,
        "max_node_load":     max_node_load,
        "mean_edge_load":    mean_edge_load,
        "max_edge_load":     max_edge_load,
        "traffic":           traffic,
        "temperature":       float(temperature),
        "wind":              float(wind),
        "aqi":               aqi,
        "disaster_type":     (active_disaster.event_type
                              if (active_disaster and active_disaster.active) else "none"),
        "disaster_zone":     (active_disaster.affected_zone
                              if (active_disaster and active_disaster.active) else -1),
        "disaster_severity": (active_disaster.severity
                              if (active_disaster and active_disaster.active) else 0.0),
        "zone_hazard":       float(zone_hazard),
        "blocked_ratio":     float(blocked_ratio),
        "evacuated_count":   evacuated_count,
        "evacuating_count":  evacuating_count,
        "evac_rate":         float(evac_rate),
        "shelter_util":      float(shelter_util),
        "avg_response_time": avg_response_time,
    })

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(logs)
df.to_csv("data/traffic_simulation.csv", index=False)
print(f"Saved {len(df)} rows → data/traffic_simulation.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df["timestep"], df["traffic"],       color="steelblue",  lw=0.8, label="Traffic")
axes[0].set_ylabel("Traffic"); axes[0].legend()

axes[1].plot(df["timestep"], df["aqi"],           color="tomato",     lw=0.8, label="AQI")
axes[1].set_ylabel("AQI"); axes[1].legend()

axes[2].plot(df["timestep"], df["zone_hazard"],   color="darkorange", lw=0.8, label="Zone hazard")
axes[2].plot(df["timestep"], df["blocked_ratio"], color="crimson",    lw=0.8, label="Blocked ratio")
axes[2].set_ylabel("Disaster"); axes[2].legend()

axes[3].plot(df["timestep"], df["evac_rate"],    color="seagreen", lw=0.8, label="Evac rate")
axes[3].plot(df["timestep"], df["shelter_util"], color="purple",   lw=0.8, label="Shelter util")
axes[3].set_ylabel("Evacuation")
axes[3].set_xlabel("Timestep (hours)")
axes[3].legend()

plt.suptitle("Digital Twin City — Disaster Simulation (routing fix applied)")
plt.tight_layout()
plt.savefig("data/traffic_simulation.png", dpi=150)
print("Plot saved → data/traffic_simulation.png")
plt.show()