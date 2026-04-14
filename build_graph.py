import os
import json
import osmnx as ox
import networkx as nx
import numpy as np

os.makedirs("data", exist_ok=True)

PLACE = "Piedmont, California, USA"
NUM_ZONES = 6       # how many zones to divide the city into
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── 1) Download road graph ───────────────────────────────────────────────────
print("Downloading road graph...")
G = ox.graph_from_place(PLACE, network_type="drive", simplify=True)

G_u = ox.convert.to_undirected(G)
largest_cc = max(nx.connected_components(G_u), key=len)
G = G.subgraph(largest_cc).copy()
print(f"Nodes: {len(G.nodes)}  Edges: {len(G.edges)}")

# ── 2) Assign zones ──────────────────────────────────────────────────────────
# Why: disasters target zones ("flood hits zone 3"), civilians evacuate from
# zones, and KPIs are reported per zone. We divide the bounding box into a
# simple grid — good enough for a simulation, no clustering needed.
#
# Each node gets a "zone" attribute (int 0..NUM_ZONES-1) based on its
# geographic position. We use a lat/lon grid split so adjacent nodes in
# space share a zone, which is what matters for disaster propagation.

node_data = dict(G.nodes(data=True))
lats = np.array([d["y"] for d in node_data.values()])
lons = np.array([d["x"] for d in node_data.values()])

lat_min, lat_max = lats.min(), lats.max()
lon_min, lon_max = lons.min(), lons.max()

# Grid: split into ceil(sqrt(NUM_ZONES)) columns and rows
cols = int(np.ceil(np.sqrt(NUM_ZONES)))
rows = int(np.ceil(NUM_ZONES / cols))

def assign_zone(lat, lon):
    col = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * cols)
    row = int((lat - lat_min) / (lat_max - lat_min + 1e-9) * rows)
    zone = min(row * cols + col, NUM_ZONES - 1)
    return int(zone)

for node, data in G.nodes(data=True):
    G.nodes[node]["zone"] = assign_zone(data["y"], data["x"])

# Count nodes per zone for population weighting later
zone_node_counts = {}
for _, d in G.nodes(data=True):
    z = d["zone"]
    zone_node_counts[z] = zone_node_counts.get(z, 0) + 1

print(f"Zones assigned: {sorted(zone_node_counts.items())}")

# ── 3) Tag infrastructure via OSM POI query ──────────────────────────────────
# Why: evacuation agents need to know where the nearest shelter is, emergency
# vehicles need fire station locations, and the improvement engine needs to
# know which zones lack hospitals. We pull real OSM data for the same place.
#
# Each infrastructure node gets snapped to its nearest road network node so
# agents can route to it using the existing graph.

print("Fetching infrastructure (hospitals, shelters, fire stations)...")

INFRA_TAGS = {
    "hospital":      {"amenity": "hospital"},
    "fire_station":  {"amenity": "fire_station"},
    "shelter":       {"amenity": ["shelter", "refuge"]},
}

infrastructure = {}   # {category: [{"node": osmid, "name": ..., "lat": ..., "lon": ..., "zone": ...}]}

for category, tags in INFRA_TAGS.items():
    try:
        gdf = ox.features_from_place(PLACE, tags=tags)
        locations = []
        for _, row in gdf.iterrows():
            # Prefer the centroid for polygons (buildings), point for nodes
            geom = row.geometry
            lat = geom.centroid.y if hasattr(geom, "centroid") else geom.y
            lon = geom.centroid.x if hasattr(geom, "centroid") else geom.x
            name = row.get("name", f"{category}_{len(locations)}")
            # Snap to nearest drivable node on the graph
            nearest = ox.distance.nearest_nodes(G, lon, lat)
            zone = G.nodes[nearest].get("zone", 0)
            locations.append({
                "name":     str(name),
                "lat":      round(lat, 6),
                "lon":      round(lon, 6),
                "node":     int(nearest),
                "zone":     int(zone),
            })
            # Tag the node itself so the graph carries the info
            G.nodes[nearest][f"is_{category}"] = True
        infrastructure[category] = locations
        print(f"  {category}: {len(locations)} found")
    except Exception as e:
        # OSM may not have data for every category in small cities — that's fine
        print(f"  {category}: not found in OSM ({e})")
        infrastructure[category] = []

# ── 4) Fallback — if OSM has no shelters, pick high-degree nodes per zone ────
# Why: small cities like Piedmont often have no tagged shelters in OSM.
# We need at least one shelter per zone for evacuation routing to work.
# High-degree nodes (busy intersections) are reasonable proxy shelter sites.

if not infrastructure["shelter"]:
    print("  No OSM shelters — assigning fallback shelter per zone...")
    fallback_shelters = []
    degree = dict(G.degree())
    for zone_id in range(NUM_ZONES):
        zone_nodes = [n for n, d in G.nodes(data=True) if d.get("zone") == zone_id]
        if not zone_nodes:
            continue
        # Pick node with highest degree in zone as the "shelter"
        best = max(zone_nodes, key=lambda n: degree.get(n, 0))
        G.nodes[best]["is_shelter"] = True
        fallback_shelters.append({
            "name": f"fallback_shelter_zone_{zone_id}",
            "lat":  round(G.nodes[best]["y"], 6),
            "lon":  round(G.nodes[best]["x"], 6),
            "node": int(best),
            "zone": int(zone_id),
        })
    infrastructure["shelter"] = fallback_shelters
    print(f"  Fallback shelters: {len(fallback_shelters)}")

# ── 4b) Fallback hospitals — if OSM has none ────────────────────────────────
if not infrastructure["hospital"]:
    print("  No OSM hospitals — assigning fallback hospitals...")
    fallback_hospitals = []
    degree = dict(G.degree())

    # pick top 2 zones by node count for better coverage
    sorted_zones = sorted(zone_node_counts.items(), key=lambda x: x[1], reverse=True)
    target_zones = [z for z, _ in sorted_zones[:2]]  # at least 2 hospitals

    for zone_id in target_zones:
        zone_nodes = [n for n, d in G.nodes(data=True) if d.get("zone") == zone_id]
        if not zone_nodes:
            continue
        best = max(zone_nodes, key=lambda n: degree.get(n, 0))
        G.nodes[best]["is_hospital"] = True
        fallback_hospitals.append({
            "name": f"fallback_hospital_zone_{zone_id}",
            "lat": round(G.nodes[best]["y"], 6),
            "lon": round(G.nodes[best]["x"], 6),
            "node": int(best),
            "zone": int(zone_id),
        })

    infrastructure["hospital"] = fallback_hospitals
    print(f"  Fallback hospitals: {len(fallback_hospitals)}")

# ── 5) Population density per zone ──────────────────────────────────────────
# Why: KPIs like "% of population evacuated in 30 min" need a population per
# zone. We approximate it from node count (denser road network = more people)
# with a small random multiplier to make zones meaningfully different.
#
# In production you'd replace this with census data.

total_nodes = sum(zone_node_counts.values())
population = {}
for zone_id in range(NUM_ZONES):
    count = zone_node_counts.get(zone_id, 1)
    base_pop = int((count / total_nodes) * 10000)   # scale to ~10k total
    noise = np.random.randint(-200, 400)
    population[zone_id] = max(100, base_pop + noise)

print(f"Estimated population per zone: {population}")

# ── 6) Save graph + metadata ─────────────────────────────────────────────────
# The graph carries zone + infrastructure tags as node attributes.
# Metadata is saved separately as JSON so simulate.py can load it without
# re-parsing the graph.

ox.save_graphml(G, "data/city.graphml")
print("Saved → data/city.graphml")

metadata = {
    "place":          PLACE,
    "num_zones":      NUM_ZONES,
    "num_nodes":      len(G.nodes),
    "num_edges":      len(G.edges),
    "bbox": {
        "lat_min": round(float(lat_min), 6),
        "lat_max": round(float(lat_max), 6),
        "lon_min": round(float(lon_min), 6),
        "lon_max": round(float(lon_max), 6),
    },
    "zone_node_counts": {int(k): v for k, v in zone_node_counts.items()},
    "population":       {int(k): v for k, v in population.items()},
    "infrastructure":   infrastructure,
}

META_PATH = "data/city_metadata.json"
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved → {META_PATH}")

# Quick summary
print("\n=== City Summary ===")
print(f"Place     : {PLACE}")
print(f"Zones     : {NUM_ZONES}")
print(f"Population: {sum(population.values()):,} (simulated)")
for cat, locs in infrastructure.items():
    print(f"{cat:<15}: {len(locs)} locations")
print("Done.")