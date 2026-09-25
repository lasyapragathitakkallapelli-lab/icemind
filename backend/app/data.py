"""
Data source simulation layer.

In production this module is replaced by:
  - Satellite imagery ingestion (Sentinel-1/2) -> CV detection (YOLO/segmentation)
  - Ocean current + wind + wave feeds (Copernicus Marine, GFS/ECMWF)
  - AIS vessel feed

For the hackathon MVP we generate a physically-plausible, deterministic
(seeded) synthetic scene around the Drake Passage / Antarctic Peninsula so the
whole pipeline - detection -> prediction -> risk -> routing -> copilot - is
fully exercised end to end without external data dependencies.
"""
import random
import time
from .models import Iceberg, LatLon, Environment, Vessel

_RNG = random.Random(42)

VESSEL = Vessel(
    name="M/V Endeavor",
    position=LatLon(lat=-63.40, lon=-59.90),
    heading_deg=195.0,
    speed_kn=13.5,
    ice_class="PC5",
)

_BASE_ENV = Environment(
    current_speed_kn=0.6,
    current_dir_deg=40.0,
    wind_speed_kn=18.0,
    wind_dir_deg=310.0,
    wave_height_m=2.1,
    sea_ice_concentration_pct=38.0,
    sea_temp_c=-1.4,
)


def _jitter_env(base: Environment, rng: random.Random) -> Environment:
    return Environment(
        current_speed_kn=max(0.05, base.current_speed_kn + rng.uniform(-0.2, 0.2)),
        current_dir_deg=(base.current_dir_deg + rng.uniform(-15, 15)) % 360,
        wind_speed_kn=max(2.0, base.wind_speed_kn + rng.uniform(-6, 6)),
        wind_dir_deg=(base.wind_dir_deg + rng.uniform(-20, 20)) % 360,
        wave_height_m=max(0.2, base.wave_height_m + rng.uniform(-0.6, 0.6)),
        sea_ice_concentration_pct=min(100, max(0, base.sea_ice_concentration_pct + rng.uniform(-10, 10))),
        sea_temp_c=base.sea_temp_c + rng.uniform(-0.5, 0.5),
    )


def _seed_icebergs():
    seeds = [
        dict(id="ICE-047", lat=-63.71, lon=-59.55, length=1.8, width=0.9, vel=0.42, hdg=32),
        dict(id="ICE-052", lat=-63.20, lon=-60.35, length=0.6, width=0.4, vel=0.31, hdg=140),
        dict(id="ICE-061", lat=-63.95, lon=-59.10, length=3.4, width=2.1, vel=0.55, hdg=15),
        dict(id="ICE-073", lat=-63.55, lon=-60.05, length=0.9, width=0.5, vel=0.28, hdg=205),
        dict(id="ICE-088", lat=-63.30, lon=-59.70, length=1.2, width=0.7, vel=0.38, hdg=60),
    ]
    icebergs = []
    for s in seeds:
        conf = round(_RNG.uniform(0.86, 0.98), 2)
        icebergs.append(Iceberg(
            id=s["id"],
            position=LatLon(lat=s["lat"], lon=s["lon"]),
            size_km=[s["length"], s["width"]],
            velocity_kn=s["vel"],
            heading_deg=s["hdg"],
            detection_confidence=conf,
            environment=_jitter_env(_BASE_ENV, _RNG),
        ))
    return icebergs


_ICEBERGS = {ib.id: ib for ib in _seed_icebergs()}
_START_TIME = time.time()

# Named Antarctic gateway ports / research stations for the Route Planner.
PORTS = [
    {"id": "ushuaia", "name": "Ushuaia, Argentina", "lat": -54.80, "lon": -68.30, "role": "source"},
    {"id": "punta-arenas", "name": "Punta Arenas, Chile", "lat": -53.16, "lon": -70.91, "role": "source"},
    {"id": "esperanza", "name": "Esperanza Station", "lat": -63.40, "lon": -56.99, "role": "dest"},
    {"id": "port-lockroy", "name": "Port Lockroy", "lat": -64.82, "lon": -63.50, "role": "dest"},
    {"id": "palmer", "name": "Palmer Station", "lat": -64.77, "lon": -64.05, "role": "dest"},
    {"id": "vernadsky", "name": "Vernadsky Station", "lat": -65.25, "lon": -64.26, "role": "dest"},
]


def get_ports():
    return PORTS


def get_port(port_id: str):
    return next((p for p in PORTS if p["id"] == port_id), None)


def environment_now() -> Environment:
    """A general 'current conditions' reading for the Command Center panel,
    independent of any single iceberg."""
    return _jitter_env(_BASE_ENV, _RNG)


def get_vessel() -> Vessel:
    return VESSEL


def list_icebergs():
    return list(_ICEBERGS.values())


def get_iceberg(iceberg_id: str):
    return _ICEBERGS.get(iceberg_id)


def tick():
    """Advance the simulated scene slightly - called by the websocket loop
    to emulate a continuously updating satellite/AIS feed."""
    elapsed_min = (time.time() - _START_TIME) / 60.0
    for ib in _ICEBERGS.values():
        # Small live jitter so the dashboard visibly updates in real time.
        ib.environment = _jitter_env(_BASE_ENV, _RNG)
    return elapsed_min
