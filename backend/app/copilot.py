"""
ICE MIND Copilot.

For the hackathon MVP this is a deterministic, template-driven "RAG-lite"
layer: it retrieves the live simulation state (vessel, icebergs, risk,
routes) and answers a handful of common navigator questions directly from
that state, so every answer is grounded and reproducible for a demo/judging
panel (no hallucination risk).

To upgrade to a real LLM+RAG copilot for production, swap `answer()`'s body
for a call to the Anthropic Messages API, passing the same retrieved context
(vessel/iceberg/risk/route JSON) as the system prompt and the navigator's
question as the user turn - the retrieval and grounding plumbing here does
not change.
"""
import os
from typing import List, Optional
from .models import Vessel, Iceberg, RiskAssessment, RouteOption, CopilotResponse
from . import risk as risk_engine


def _most_relevant_hazard(vessel: Vessel, icebergs: List[Iceberg]) -> Optional[RiskAssessment]:
    if not icebergs:
        return None
    assessments = [risk_engine.assess_iceberg_risk(vessel, ib) for ib in icebergs]
    assessments.sort(key=lambda a: a.score, reverse=True)
    return assessments[0]


def answer(question: str, vessel: Vessel, icebergs: List[Iceberg],
           routes: Optional[List[RouteOption]] = None) -> CopilotResponse:
    q = question.lower()
    top = _most_relevant_hazard(vessel, icebergs)

    if any(k in q for k in ["why", "explain"]) or "safer" in q:
        if routes:
            rec = next((r for r in routes if r.recommended), routes[0])
            base = routes[0]
            delta_dist = (rec.distance_km - base.distance_km) / base.distance_km * 100 if base.distance_km else 0
            delta_risk = (base.risk_score - rec.risk_score)
            txt = (f"Route '{rec.label}' is recommended: {abs(delta_dist):.0f}% "
                   f"{'longer' if delta_dist >= 0 else 'shorter'} distance for a "
                   f"{delta_risk:.0f}-point reduction in predicted navigation risk "
                   f"({base.risk_score} -> {rec.risk_score}).")
            return CopilotResponse(answer=txt, risk_before=base.risk_score, risk_after=rec.risk_score)
        if top:
            factor_txt = ", ".join(f"{f.label} ({f.impact})" for f in top.factors if f.impact != "Low")
            return CopilotResponse(answer=f"Primary risk drivers right now: {factor_txt or 'no major factors'}.",
                                    risk_before=top.score)

    if any(k in q for k in ["is my route safe", "24 hour", "next 24", "route safe", " safe"]):
        if top is None:
            return CopilotResponse(answer="No tracked icebergs currently threaten the planned route.")
        if top.band in ("HIGH", "CRITICAL"):
            eta = top.time_to_encounter_hours
            txt = (f"Caution: your route intersects the predicted uncertainty corridor of "
                   f"{top.iceberg_id} in approximately {eta:.0f}h. Current risk: {top.score}/100 "
                   f"({top.band}). Recommended action: request a safer route or alter heading.")
            return CopilotResponse(answer=txt, risk_before=top.score,
                                    recommended_action="Shift heading east / request AI route")
        return CopilotResponse(answer=f"Route currently looks {top.band.lower()} risk "
                                       f"({top.score}/100), closest tracked hazard is {top.iceberg_id}.",
                                risk_before=top.score)

    if "what if" in q or "reduce speed" in q or "slow" in q:
        return CopilotResponse(
            answer="Try the What-If Simulator panel: reducing speed or altering heading recomputes "
                   "risk live against the same predicted iceberg trajectories.",
        )

    if any(k in q for k in ["nearest", "closest", "hazard"]):
        if top is None:
            return CopilotResponse(answer="No significant hazards currently tracked near the vessel.")
        return CopilotResponse(
            answer=f"Nearest significant hazard: {top.iceberg_id}, risk {top.score}/100 ({top.band}), "
                   f"estimated time to closest approach ~{(top.time_to_encounter_hours or 0):.1f}h.",
            risk_before=top.score,
        )

    # Fallback grounded summary.
    if top:
        return CopilotResponse(
            answer=f"Current situation: {len(icebergs)} icebergs tracked, highest risk is "
                   f"{top.iceberg_id} at {top.score}/100 ({top.band}). Ask me about route safety, "
                   f"why a route is recommended, or nearby hazards.",
            risk_before=top.score,
        )
    return CopilotResponse(answer="All tracked corridors currently show low risk.")
