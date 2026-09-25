"""
Dynamic Navigation Risk Engine.

Computes a continuous 0-100 risk score per iceberg (and an aggregate route
risk) instead of a binary safe/dangerous classification, and always returns
the weighted factor breakdown so the frontend / copilot can explain *why*
a score is what it is (explainable AI requirement).
"""
import math
from typing import List, Optional
from .models import Iceberg, Vessel, RiskAssessment, RiskFactor, RiskTimelinePoint
from .prediction import position_at, _uncertainty_radius_km, _confidence

EARTH_RADIUS_KM = 6371.0
KN_TO_KMH = 1.852


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _vessel_position_at(vessel: Vessel, hours: float):
    dist_km = vessel.speed_kn * KN_TO_KMH * hours
    rad = math.radians(vessel.heading_deg)
    east = dist_km * math.sin(rad)
    north = dist_km * math.cos(rad)
    d_lat = north / EARTH_RADIUS_KM * (180 / math.pi)
    d_lon = east / (EARTH_RADIUS_KM * math.cos(math.radians(vessel.position.lat))) * (180 / math.pi)
    return vessel.position.lat + d_lat, vessel.position.lon + d_lon


def _closest_point_of_approach(vessel: Vessel, iceberg: Iceberg, horizon_hours: float = 48, steps: int = 96):
    best = (None, float("inf"))
    for i in range(steps + 1):
        h = horizon_hours * i / steps
        vlat, vlon = _vessel_position_at(vessel, h)
        ipos = position_at(iceberg, h)
        d = haversine_km(vlat, vlon, ipos.lat, ipos.lon)
        # Treat the uncertainty radius as effectively widening the hazard.
        effective = d - _uncertainty_radius_km(h, iceberg) * 0.5
        if effective < best[1]:
            best = (h, effective)
    return best  # (time_to_cpa_hours, effective_distance_km)


def _weather_severity(iceberg: Iceberg) -> float:
    env = iceberg.environment
    # 0-1 scale from wind + wave, capped.
    return min(1.0, (env.wind_speed_kn / 45) * 0.5 + (env.wave_height_m / 6) * 0.5)


def assess_iceberg_risk(vessel: Vessel, iceberg: Iceberg) -> RiskAssessment:
    time_to_cpa, eff_dist = _closest_point_of_approach(vessel, iceberg)

    # Proximity factor: closer (or negative = inside the uncertainty cone) -> higher risk.
    proximity = max(0.0, min(1.0, 1 - (eff_dist / 40.0))) if eff_dist is not None else 0.0

    length_km = iceberg.size_km[0]
    size_factor = min(1.0, length_km / 4.0)

    ice_conc_factor = iceberg.environment.sea_ice_concentration_pct / 100

    weather_factor = _weather_severity(iceberg)

    uncertainty_factor = 1 - _confidence(time_to_cpa or 48, iceberg)

    speed_factor = min(1.0, vessel.speed_kn / 18.0)

    weights = {
        "proximity": 0.32,
        "uncertainty": 0.18,
        "size": 0.14,
        "ice_conc": 0.14,
        "weather": 0.12,
        "speed": 0.10,
    }
    values = {
        "proximity": proximity,
        "uncertainty": uncertainty_factor,
        "size": size_factor,
        "ice_conc": ice_conc_factor,
        "weather": weather_factor,
        "speed": speed_factor,
    }
    score = sum(weights[k] * values[k] for k in weights) * 100
    score = round(min(100, max(0, score)), 1)

    if score >= 80:
        band = "CRITICAL"
    elif score >= 55:
        band = "HIGH"
    elif score >= 30:
        band = "MODERATE"
    else:
        band = "LOW"

    def impact(v):
        return "High" if v >= 0.6 else "Moderate" if v >= 0.3 else "Low"

    factors = [
        RiskFactor(label="Iceberg proximity / trajectory intersection", impact=impact(proximity),
                   weight=weights["proximity"], contribution=round(weights["proximity"] * proximity * 100, 1)),
        RiskFactor(label="Prediction uncertainty", impact=impact(uncertainty_factor),
                   weight=weights["uncertainty"], contribution=round(weights["uncertainty"] * uncertainty_factor * 100, 1)),
        RiskFactor(label="Iceberg size", impact=impact(size_factor),
                   weight=weights["size"], contribution=round(weights["size"] * size_factor * 100, 1)),
        RiskFactor(label="Sea-ice concentration", impact=impact(ice_conc_factor),
                   weight=weights["ice_conc"], contribution=round(weights["ice_conc"] * ice_conc_factor * 100, 1)),
        RiskFactor(label="Weather severity (wind/wave)", impact=impact(weather_factor),
                   weight=weights["weather"], contribution=round(weights["weather"] * weather_factor * 100, 1)),
        RiskFactor(label="Vessel speed", impact=impact(speed_factor),
                   weight=weights["speed"], contribution=round(weights["speed"] * speed_factor * 100, 1)),
    ]

    if band in ("HIGH", "CRITICAL") and time_to_cpa is not None:
        narrative = (f"Iceberg {iceberg.id} predicted to enter the vessel's effective corridor in "
                     f"~{time_to_cpa:.1f}h (closest approach ~{max(eff_dist,0):.1f} km net of uncertainty).")
    else:
        narrative = f"Iceberg {iceberg.id} poses {band.lower()} risk to the current route over the next 48h."

    return RiskAssessment(
        iceberg_id=iceberg.id,
        score=score,
        band=band,
        time_to_encounter_hours=round(time_to_cpa, 2) if time_to_cpa is not None else None,
        factors=factors,
        narrative=narrative,
    )


def assess_route_risk(vessel: Vessel, icebergs: List[Iceberg]) -> RiskAssessment:
    """Aggregate risk = risk from the single most dangerous iceberg, with a
    modest contribution from the others (concurrent hazards compound)."""
    assessments = [assess_iceberg_risk(vessel, ib) for ib in icebergs]
    if not assessments:
        return RiskAssessment(score=0, band="LOW", factors=[], narrative="No tracked icebergs near the route.")
    assessments.sort(key=lambda a: a.score, reverse=True)
    top = assessments[0]
    others_bonus = sum(a.score for a in assessments[1:]) * 0.05
    score = round(min(100, top.score + others_bonus), 1)
    band = top.band if score < 80 else "CRITICAL"
    return RiskAssessment(
        iceberg_id=top.iceberg_id,
        score=score,
        band=band,
        time_to_encounter_hours=top.time_to_encounter_hours,
        factors=top.factors,
        narrative=top.narrative,
    )


def risk_timeline(vessel: Vessel, icebergs: List[Iceberg], horizons=(0, 6, 12, 18, 24)) -> List[RiskTimelinePoint]:
    points = []
    for h in horizons:
        # Project the vessel forward and score against icebergs' predicted positions at h.
        vlat, vlon = _vessel_position_at(vessel, h)
        worst = 0.0
        worst_band = "LOW"
        for ib in icebergs:
            ipos = position_at(ib, h)
            d = haversine_km(vlat, vlon, ipos.lat, ipos.lon)
            eff = max(0.0, d - _uncertainty_radius_km(h, ib) * 0.5)
            proximity = max(0.0, min(1.0, 1 - eff / 40.0))
            size_factor = min(1.0, ib.size_km[0] / 4.0)
            unc = 1 - _confidence(h, ib)
            s = (0.5 * proximity + 0.2 * unc + 0.3 * size_factor) * 100
            if s > worst:
                worst = s
                worst_band = "CRITICAL" if s >= 80 else "HIGH" if s >= 55 else "MODERATE" if s >= 30 else "LOW"
        points.append(RiskTimelinePoint(hours_ahead=h, score=round(worst, 1), band=worst_band))
    return points
