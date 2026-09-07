"""
Graph Theory Isolation Engine for North Eastern Region (NER).
Models transportation arteries (NH-06, NH-10, NH-29, SH networks) as a
network graph and calculates the Village Isolation Index (VII) when
landslide blockades sever roads and bridges.
"""

import networkx as nx
from typing import List, Dict, Any

class RegionalRoadGraphEngine:
    def __init__(self):
        self.G = nx.Graph()
        self._load_default_ner_network()

    def _load_default_ner_network(self):
        """
        Loads representative arterial network graph for key corridors:
        1. NH-06 Umiam-Shillong-Dawki-Pynursla Corridor (Meghalaya)
        2. Dibang Valley Corridor (Arunachal Pradesh)
        3. NH-10 Gangtok-Teesta Corridor (Sikkim)
        """
        self.G.clear()

        # Nodes (Villages, Hospitals, Junctions, Bridges)
        nodes = [
            # Meghalaya NH-06 Corridor
            {"id": "V_PYNURSLA", "type": "Village", "name": "Pynursla", "population": 4500, "vulnerable": 850, "lat": 25.3082, "lon": 91.9002},
            {"id": "V_DAWKI", "type": "Village", "name": "Dawki Border", "population": 2800, "vulnerable": 520, "lat": 25.1834, "lon": 92.0197},
            {"id": "V_NONGRIAT", "type": "Village", "name": "Nongriat", "population": 950, "vulnerable": 240, "lat": 25.2505, "lon": 91.6702},
            {"id": "V_MAWLYNNONG", "type": "Village", "name": "Mawlynnong", "population": 1200, "vulnerable": 310, "lat": 25.2016, "lon": 91.9161},
            
            # Dibang Valley Corridor (Arunachal Pradesh)
            {"id": "V_ANINI", "type": "Village", "name": "Anini Town", "population": 4200, "vulnerable": 900, "lat": 28.7900, "lon": 95.9000},
            {"id": "V_ETALIN", "type": "Village", "name": "Etalin Hamlet", "population": 1100, "vulnerable": 280, "lat": 28.6100, "lon": 95.8400},
            {"id": "V_MALINEY", "type": "Village", "name": "Maliney Village", "population": 750, "vulnerable": 190, "lat": 28.5300, "lon": 95.8100},

            # Junctions and Critical Bridges
            {"id": "BRG_UMIAM", "type": "Bridge", "name": "Umiam Arterial Bridge", "lat": 25.6601, "lon": 91.9125},
            {"id": "BRG_WAHREW", "type": "Bridge", "name": "Wahrew Arch Bridge", "lat": 25.2300, "lon": 91.8900},
            {"id": "BRG_DIBANG_SPAN", "type": "Bridge", "name": "Dibang River Suspension Bridge", "lat": 28.5800, "lon": 95.8200},
            {"id": "JNC_UMSNING", "type": "RoadJunction", "name": "Umsning Bypass Junction", "lat": 25.7523, "lon": 91.8891},
            {"id": "JNC_ROING_FORK", "type": "RoadJunction", "name": "Roing Valley Fork", "lat": 28.1400, "lon": 95.8300},

            # Tertiary Medical Centers & Designated Shelters
            {"id": "HOSP_NEIGRIHMS", "type": "Hospital", "name": "NEIGRIHMS Tertiary Care (Shillong)", "beds": 500, "lat": 25.5991, "lon": 91.9392},
            {"id": "HOSP_ROING_DIST", "type": "Hospital", "name": "Roing District Civil Hospital", "beds": 150, "lat": 28.1300, "lon": 95.8400},
            {"id": "SHELTER_NONGPOH", "type": "ReliefShelter", "name": "Nongpoh Community Shelter", "capacity": 1200, "lat": 25.9015, "lon": 91.8791},
            {"id": "SHELTER_ANINI_GYM", "type": "ReliefShelter", "name": "Anini Indoor Stadium Shelter", "capacity": 800, "lat": 28.7950, "lon": 95.8950}
        ]

        for n in nodes:
            self.G.add_node(n["id"], **n)

        # Edges (Road Segments)
        edges = [
            # Meghalaya Network
            {"u": "V_DAWKI", "v": "BRG_WAHREW", "road_id": "RD_ML_01", "name": "Dawki-Wahrew Road", "length_km": 12.4},
            {"u": "V_MAWLYNNONG", "v": "BRG_WAHREW", "road_id": "RD_ML_02", "name": "Mawlynnong Link", "length_km": 8.1},
            {"u": "BRG_WAHREW", "v": "V_PYNURSLA", "road_id": "RD_ML_03", "name": "Wahrew-Pynursla Ridge", "length_km": 15.6},
            {"u": "V_NONGRIAT", "v": "V_PYNURSLA", "road_id": "RD_ML_04", "name": "Sohra-Pynursla Trail", "length_km": 18.0},
            {"u": "V_PYNURSLA", "v": "BRG_UMIAM", "road_id": "RD_ML_05", "name": "NH-06 South Segment", "length_km": 34.0},
            {"u": "BRG_UMIAM", "v": "HOSP_NEIGRIHMS", "road_id": "RD_ML_06", "name": "NEIGRIHMS Expressway Link", "length_km": 9.5},
            {"u": "BRG_UMIAM", "v": "JNC_UMSNING", "road_id": "RD_ML_07", "name": "NH-06 Northbound Bypass", "length_km": 14.2},
            {"u": "JNC_UMSNING", "v": "SHELTER_NONGPOH", "road_id": "RD_ML_08", "name": "Nongpoh Shelter Access", "length_km": 16.0},

            # Arunachal Pradesh Dibang Valley Network
            {"u": "V_ANINI", "v": "V_ETALIN", "road_id": "RD_AR_01", "name": "Anini-Etalin Highway", "length_km": 42.0},
            {"u": "V_ETALIN", "v": "BRG_DIBANG_SPAN", "road_id": "RD_AR_02", "name": "Etalin-Dibang Gorge Road", "length_km": 19.5},
            {"u": "V_MALINEY", "v": "BRG_DIBANG_SPAN", "road_id": "RD_AR_03", "name": "Maliney Feeder", "length_km": 7.3},
            {"u": "BRG_DIBANG_SPAN", "v": "JNC_ROING_FORK", "road_id": "RD_AR_04", "name": "Lower Dibang Pass", "length_km": 54.0},
            {"u": "JNC_ROING_FORK", "v": "HOSP_ROING_DIST", "road_id": "RD_AR_05", "name": "Roing Hospital Link", "length_km": 3.8},
            {"u": "V_ANINI", "v": "SHELTER_ANINI_GYM", "road_id": "RD_AR_06", "name": "Anini Local Shelter Road", "length_km": 2.0}
        ]

        for e in edges:
            self.G.add_edge(e["u"], e["v"], road_id=e["road_id"], name=e["name"], length_km=e["length_km"], is_blocked=False)

    def get_network_state(self) -> Dict[str, Any]:
        """Returns node and edge dictionaries for GIS visualization."""
        nodes_list = []
        for n, data in self.G.nodes(data=True):
            node_info = dict(data)
            node_info["id"] = n
            nodes_list.append(node_info)

        edges_list = []
        for u, v, data in self.G.edges(data=True):
            edge_info = dict(data)
            edge_info["source"] = u
            edge_info["target"] = v
            edges_list.append(edge_info)

        return {"nodes": nodes_list, "edges": edges_list}

    def calculate_isolation_index(self, blocked_road_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Calculates the Village Isolation Index (VII) and prioritized evacuation ranking.
        """
        if blocked_road_ids is None:
            blocked_road_ids = []

        # Create perturbed network G' by removing blocked segments
        G_prime = self.G.copy()
        edges_to_remove = []
        for u, v, data in G_prime.edges(data=True):
            if data.get("road_id") in blocked_road_ids:
                edges_to_remove.append((u, v))

        G_prime.remove_edges_from(edges_to_remove)

        hospital_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") == "Hospital"]
        village_nodes = [(n, d) for n, d in self.G.nodes(data=True) if d.get("type") == "Village"]

        # Articulation points (Cut-vertices)
        cut_nodes = set(nx.articulation_points(self.G))

        rankings = []
        for v_id, v_data in village_nodes:
            # Baseline distance to closest hospital
            d_base_list = []
            for h in hospital_nodes:
                try:
                    d = nx.shortest_path_length(self.G, source=v_id, target=h, weight="length_km")
                    d_base_list.append(d)
                except nx.NetworkXNoPath:
                    pass
            d_base = min(d_base_list) if d_base_list else 15.0

            # Post-failure distance in G'
            d_post_list = []
            for h in hospital_nodes:
                try:
                    d = nx.shortest_path_length(G_prime, source=v_id, target=h, weight="length_km")
                    d_post_list.append(d)
                except nx.NetworkXNoPath:
                    pass

            if not d_post_list:
                i_hosp = 1.0 # 100% Severed
                detour_factor = 3.0
                post_dist = None
                is_isolated = True
            else:
                i_hosp = 0.0
                post_dist = min(d_post_list)
                detour_factor = max((post_dist - d_base) / d_base, 0.0)
                is_isolated = False

            pop_total = v_data.get("population", 1000)
            pop_vuln = v_data.get("vulnerable", 200)
            vuln_ratio = pop_vuln / pop_total
            i_single = 1.0 if v_id in cut_nodes else 0.0

            # Village Isolation Index formulation
            vii = (
                0.45 * i_hosp +
                0.25 * min(detour_factor, 3.0) / 3.0 +
                0.20 * vuln_ratio +
                0.10 * i_single
            )

            rankings.append({
                "village_id": v_id,
                "village_name": v_data["name"],
                "latitude": v_data["lat"],
                "longitude": v_data["lon"],
                "vii_score": round(float(vii), 3),
                "is_severed": is_isolated,
                "baseline_km": round(d_base, 1),
                "post_collapse_km": round(post_dist, 1) if post_dist else "TRAPPED (NO ROUTE)",
                "population": pop_total,
                "vulnerable_citizens": pop_vuln,
                "status_badge": "AIRDROP MANDATED" if vii >= 0.65 else ("HIGH DETOUR" if is_isolated or detour_factor > 0.5 else "OPERATIONAL")
            })

        rankings.sort(key=lambda x: x["vii_score"], reverse=True)
        return rankings
