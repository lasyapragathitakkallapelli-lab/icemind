"""
ICE MIND backend.

Pipeline exposed via HTTP/WebSocket:

  data.py (satellite/AIS/env feed simulation)
        -> prediction.py (physics-guided trajectory + uncertainty)
        -> risk.py (dynamic navigation risk engine, explainable)
        -> routing.py (risk-weighted A* route optimizer)
        -> copilot.py (grounded navigator Q&A)

Run:
    uvicorn app.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""
import asyncio
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from . import data, prediction, risk as risk_engine, routing, copilot
from .models import (
    Iceberg, TrajectoryPrediction, RiskAssessment, RouteRequest, RouteOption,
    WhatIfRequest, CopilotQuery, CopilotResponse, Vessel, RiskTimelinePoint,
    Environment, LatLon,
)


class PortRouteRequest(BaseModel):
    source_id: str
    dest_id: str
    max_speed_kn: float = 14.0

app = FastAPI(title="ICE MIND", version="0.1.0",
              description="Predictive maritime intelligence for Antarctic iceberg navigation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/vessel", response_model=Vessel)
def get_vessel():
    return data.get_vessel()


@app.get("/api/icebergs", response_model=List[Iceberg])
def get_icebergs():
    return data.list_icebergs()


@app.get("/api/icebergs/{iceberg_id}", response_model=Iceberg)
def get_iceberg(iceberg_id: str):
    ib = data.get_iceberg(iceberg_id)
    if not ib:
        raise HTTPException(404, "iceberg not found")
    return ib


@app.get("/api/icebergs/{iceberg_id}/predict", response_model=TrajectoryPrediction)
def predict(iceberg_id: str):
    ib = data.get_iceberg(iceberg_id)
    if not ib:
        raise HTTPException(404, "iceberg not found")
    return prediction.predict_trajectory(ib)


@app.get("/api/icebergs/{iceberg_id}/risk", response_model=RiskAssessment)
def iceberg_risk(iceberg_id: str):
    ib = data.get_iceberg(iceberg_id)
    if not ib:
        raise HTTPException(404, "iceberg not found")
    return risk_engine.assess_iceberg_risk(data.get_vessel(), ib)


@app.get("/api/risk/route", response_model=RiskAssessment)
def route_risk():
    return risk_engine.assess_route_risk(data.get_vessel(), data.list_icebergs())


@app.get("/api/risk/timeline", response_model=List[RiskTimelinePoint])
def timeline():
    return risk_engine.risk_timeline(data.get_vessel(), data.list_icebergs())


@app.post("/api/route/optimize", response_model=List[RouteOption])
def optimize(req: RouteRequest):
    vessel = data.get_vessel().model_copy(update={"speed_kn": req.max_speed_kn})
    options = routing.optimize_routes(req.start, req.destination, vessel, data.list_icebergs())
    if not options:
        raise HTTPException(500, "no feasible route found")
    return options


@app.get("/api/ports")
def ports():
    return data.get_ports()


@app.get("/api/environment", response_model=Environment)
def environment():
    return data.environment_now()


@app.post("/api/routes", response_model=List[RouteOption])
def routes_between_ports(req: PortRouteRequest):
    src = data.get_port(req.source_id)
    dest = data.get_port(req.dest_id)
    if not src or not dest:
        raise HTTPException(404, "unknown port id")
    vessel = data.get_vessel().model_copy(update={"speed_kn": req.max_speed_kn})
    start = LatLon(lat=src["lat"], lon=src["lon"])
    destination = LatLon(lat=dest["lat"], lon=dest["lon"])
    options = routing.optimize_routes(start, destination, vessel, data.list_icebergs())
    if not options:
        raise HTTPException(500, "no feasible route found")
    return options


@app.post("/api/whatif", response_model=RiskAssessment)
def what_if(req: WhatIfRequest):
    vessel = data.get_vessel().model_copy(update={
        "speed_kn": req.speed_kn,
        "heading_deg": req.heading_deg,
    })
    return risk_engine.assess_route_risk(vessel, data.list_icebergs())


@app.post("/api/copilot", response_model=CopilotResponse)
def ask_copilot(q: CopilotQuery):
    vessel = data.get_vessel()
    icebergs = data.list_icebergs()
    routes = None
    try:
        routes = routing.optimize_routes(
            vessel.position,
            data.list_icebergs()[0].position,  # placeholder destination for "why" questions
            vessel, icebergs,
        )
    except Exception:
        routes = None
    return copilot.answer(q.question, vessel, icebergs, routes)


@app.get("/api/emergency")
def emergency():
    """One-button emergency navigation: nearest hazard, safe heading, ETA to hazard."""
    vessel = data.get_vessel()
    icebergs = data.list_icebergs()
    assessment = risk_engine.assess_route_risk(vessel, icebergs)
    ib = data.get_iceberg(assessment.iceberg_id) if assessment.iceberg_id else None

    safe_heading = (vessel.heading_deg + 35) % 360  # simple avoidance heuristic for the demo
    reduced_speed = max(6.0, vessel.speed_kn - 4)
    safer_vessel = vessel.model_copy(update={"heading_deg": safe_heading, "speed_kn": reduced_speed})
    after = risk_engine.assess_route_risk(safer_vessel, icebergs)

    return {
        "hazard": ib,
        "time_to_encounter_hours": assessment.time_to_encounter_hours,
        "recommended_heading_deg": round(safe_heading, 1),
        "recommended_speed_kn": reduced_speed,
        "risk_before": assessment.score,
        "risk_after": after.score,
    }


@app.websocket("/ws/live")
async def live_feed(ws: WebSocket):
    """Streams updated icebergs + route risk every 3s so the map/risk panel
    update without polling (real-time backend requirement)."""
    await ws.accept()
    try:
        while True:
            data.tick()
            vessel = data.get_vessel()
            icebergs = data.list_icebergs()
            payload = {
                "vessel": vessel.model_dump(),
                "icebergs": [ib.model_dump() for ib in icebergs],
                "route_risk": risk_engine.assess_route_risk(vessel, icebergs).model_dump(),
            }
            await ws.send_json(payload)
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
