"""
Spatiotemporal / physics-guided trajectory prediction.

Design note (for the pitch): a production system replaces the vector-combine
step below with a ConvLSTM / Temporal Fusion Transformer trained on historical
drift tracks, using this same physics vector as one of its input features
("physics-guided machine learning" - the physics term anchors the learned
correction so the model can't diverge wildly on sparse training data). The
interface (`predict_trajectory`) is written so swapping in a trained model
only touches this file.

Drift model:
    v_iceberg = w_wind * (wind_factor * wind_vector)
              + w_current * current_vector
              + w_persist * observed_velocity_vector

Uncertainty grows with the forecast horizon (icebergs are increasingly hard to
pin down the further out you predict) and confidence decays accordingly -
this directly implements the "uncertainty-aware AI" requirement rather than
pretending the model is perfect at 48h.
"""
import math
from typing import List
from .models import Iceberg, LatLon, TrajectoryPoint, TrajectoryPrediction

EARTH_RADIUS_KM = 6371.0
KN_TO_KMH = 1.852

# Icebergs are driven mostly by subsurface current + their own momentum;
# wind affects the small fraction of freeboard exposed above water.
WIND_DRIFT_FACTOR = 0.02      # ~2% of wind speed, classic iceberg drift heuristic
W_WIND = 0.15
W_CURRENT = 0.35
W_PERSIST = 0.50               # observed velocity already encodes net forces


def _vector(speed_kmh: float, bearing_deg: float):
    rad = math.radians(bearing_deg)
    return speed_kmh * math.sin(rad), speed_kmh * math.cos(rad)  # (east, north) km/h


def _bearing_speed(east: float, north: float):
    speed = math.hypot(east, north)
    bearing = (math.degrees(math.atan2(east, north))) % 360
    return speed, bearing


def _displace(origin: LatLon, east_km: float, north_km: float) -> LatLon:
    d_lat = north_km / EARTH_RADIUS_KM * (180 / math.pi)
    d_lon = east_km / (EARTH_RADIUS_KM * math.cos(math.radians(origin.lat))) * (180 / math.pi)
    return LatLon(lat=origin.lat + d_lat, lon=origin.lon + d_lon)


def _net_drift_vector_kmh(iceberg: Iceberg):
    env = iceberg.environment
    wind_e, wind_n = _vector(env.wind_speed_kn * KN_TO_KMH * WIND_DRIFT_FACTOR, env.wind_dir_deg)
    cur_e, cur_n = _vector(env.current_speed_kn * KN_TO_KMH, env.current_dir_deg)
    obs_e, obs_n = _vector(iceberg.velocity_kn * KN_TO_KMH, iceberg.heading_deg)

    east = W_WIND * wind_e + W_CURRENT * cur_e + W_PERSIST * obs_e
    north = W_WIND * wind_n + W_CURRENT * cur_n + W_PERSIST * obs_n
    return east, north


def _uncertainty_radius_km(hours: float, iceberg: Iceberg) -> float:
    # Base uncertainty scales with size (larger bergs are tracked more
    # reliably) and grows super-linearly with the forecast horizon.
    base = 0.4 + 0.3 * (1.0 - iceberg.detection_confidence) * 5
    growth = 0.35 * (hours ** 1.25)
    ice_conc_penalty = iceberg.environment.sea_ice_concentration_pct / 100 * 0.5 * hours
    return round(base + growth + ice_conc_penalty, 2)


def _confidence(hours: float, iceberg: Iceberg) -> float:
    decay = 0.018 * hours + 0.004 * hours ** 1.4
    conf = iceberg.detection_confidence * math.exp(-decay)
    return round(max(0.04, min(0.99, conf)), 3)


def predict_trajectory(iceberg: Iceberg, horizons_hours: List[float] = None) -> TrajectoryPrediction:
    horizons_hours = horizons_hours or [6, 12, 24, 48]
    east_kmh, north_kmh = _net_drift_vector_kmh(iceberg)

    points = []
    for h in horizons_hours:
        pos = _displace(iceberg.position, east_kmh * h, north_kmh * h)
        points.append(TrajectoryPoint(
            hours_ahead=h,
            position=pos,
            confidence=_confidence(h, iceberg),
            uncertainty_radius_km=_uncertainty_radius_km(h, iceberg),
        ))

    return TrajectoryPrediction(iceberg_id=iceberg.id, points=points)


def position_at(iceberg: Iceberg, hours: float) -> LatLon:
    east_kmh, north_kmh = _net_drift_vector_kmh(iceberg)
    return _displace(iceberg.position, east_kmh * hours, north_kmh * hours)
