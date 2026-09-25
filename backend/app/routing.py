"""
Route optimisation.

Formulated as a shortest-path search over a lat/lon grid where each edge cost
is:

    cost(edge) = distance_km(edge) + risk_weight * hazard(edge_midpoint, eta)

Hazard at a grid node is a smooth (Gaussian-falloff) field built from every
tracked iceberg's *predicted* position at the estimated arrival time at that
node - so the optimizer is routing around where ice will be, not just where
it is now. Running the same A* search with three different risk_weight
values produces the "Shortest / Recommended / Conservative" family the pitch
deck describes, which is a lightweight stand-in for a full multi-objective
(distance, risk, fuel) optimizer.
"""
import heapq
import math
from typing import List
import numpy as np

from .models import Iceberg, LatLon, RouteOption, Vessel
from .risk import haversine_km
from .prediction import position_at

GRID_N = 22
KN_TO_KMH = 1.852


def _build_grid(start: LatLon, dest: LatLon):
    pad_lat = max(0.4, abs(dest.lat - start.lat) * 0.25)
    pad_lon = max(0.4, abs(dest.lon - start.lon) * 0.25)
    lat_min, lat_max = sorted([start.lat, dest.lat])
    lon_min, lon_max = sorted([start.lon, dest.lon])
    lat_min -= pad_lat; lat_max += pad_lat
    lon_min -= pad_lon; lon_max += pad_lon

    lats = np.linspace(lat_min, lat_max, GRID_N)
    lons = np.linspace(lon_min, lon_max, GRID_N)
    return lats, lons


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _hazard_field(lats, lons, icebergs: List[Iceberg], nominal_eta_hours: float, start: LatLon, dest: LatLon):
    """hazard[i, j] in ~[0, 100], evaluated at the estimated time the vessel
    would reach that node if travelling roughly straight-line."""
    total_dist = haversine_km(start.lat, start.lon, dest.lat, dest.lon) or 1.0
    hazard = np.zeros((len(lats), len(lons)))
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            progress = haversine_km(start.lat, start.lon, la, lo) / total_dist
            progress = min(1.0, max(0.0, progress))
            eta_h = progress * nominal_eta_hours
            node_hazard = 0.0
            for ib in icebergs:
                pos = position_at(ib, eta_h)
                d = haversine_km(la, lo, pos.lat, pos.lon)
                size_km = max(ib.size_km)
                sigma = 6 + size_km * 3  # bigger bergs cast a wider hazard shadow
                intensity = 55 + 35 * size_km / 4.0
                node_hazard += intensity * math.exp(-(d ** 2) / (2 * sigma ** 2))
            hazard[i, j] = min(100.0, node_hazard)
    return hazard


def _astar(lats, lons, hazard, start_idx, goal_idx, risk_weight):
    def node_latlon(idx):
        i, j = idx
        return lats[i], lons[j]

    def h(idx):
        la, lo = node_latlon(idx)
        gla, glo = node_latlon(goal_idx)
        return haversine_km(la, lo, gla, glo)

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    open_set = [(h(start_idx), 0.0, start_idx, None)]
    came_from = {}
    g_score = {start_idx: 0.0}
    visited = set()

    while open_set:
        _, g, current, parent = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent
        if current == goal_idx:
            break
        ci, cj = current
        for di, dj in neighbors:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < len(lats) and 0 <= nj < len(lons)):
                continue
            nxt = (ni, nj)
            la1, lo1 = node_latlon(current)
            la2, lo2 = node_latlon(nxt)
            dist = haversine_km(la1, lo1, la2, lo2)
            mid_hazard = (hazard[ci, cj] + hazard[ni, nj]) / 2
            cost = dist + risk_weight * mid_hazard
            tentative_g = g + cost
            if tentative_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative_g
                heapq.heappush(open_set, (tentative_g + h(nxt), tentative_g, nxt, current))

    if goal_idx not in came_from:
        return None
    path = []
    node = goal_idx
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return path


def _path_metrics(path, lats, lons, hazard, vessel_speed_kn):
    waypoints = [LatLon(lat=lats[i], lon=lons[j]) for i, j in path]
    dist = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        dist += haversine_km(a.lat, a.lon, b.lat, b.lon)
    eta_hours = dist / (vessel_speed_kn * KN_TO_KMH / 1.852)  # speed already in knots -> km/h via *1.852
    eta_hours = dist / (vessel_speed_kn * 1.852)
    path_hazards = [hazard[i, j] for i, j in path]
    risk_score = round(min(100.0, max(path_hazards) * 0.9 + (sum(path_hazards) / len(path_hazards)) * 0.1), 1)
    return waypoints, round(dist, 1), round(eta_hours, 1), risk_score


def optimize_routes(start: LatLon, dest: LatLon, vessel: Vessel, icebergs: List[Iceberg]) -> List[RouteOption]:
    lats, lons = _build_grid(start, dest)
    start_idx = (_nearest_index(lats, start.lat), _nearest_index(lons, start.lon))
    goal_idx = (_nearest_index(lats, dest.lat), _nearest_index(lons, dest.lon))

    direct_km = haversine_km(start.lat, start.lon, dest.lat, dest.lon)
    nominal_eta = direct_km / (vessel.speed_kn * 1.852)
    hazard = _hazard_field(lats, lons, icebergs, nominal_eta, start, dest)

    variants = [
        ("shortest", "Shortest", 0.03),
        ("recommended", "Recommended", 0.55),
        ("conservative", "Conservative", 1.6),
    ]

    options = []
    best_score_idx = None
    for vid, label, weight in variants:
        path = _astar(lats, lons, hazard, start_idx, goal_idx, weight)
        if path is None:
            continue
        waypoints, dist, eta, risk = _path_metrics(path, lats, lons, hazard, vessel.speed_kn)
        options.append(RouteOption(id=vid, label=label, waypoints=waypoints,
                                    distance_km=dist, eta_hours=eta, risk_score=risk))

    if options:
        # Recommend the route with the best (distance, risk) trade-off:
        # lowest risk unless the "shortest" option is already low risk.
        min_risk = min(o.risk_score for o in options)
        for o in options:
            if o.risk_score == min_risk:
                o.recommended = True
                break

    return options
