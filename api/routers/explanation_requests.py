"""Authenticated private intent around shared lecture processing."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from api.auth import get_current_user
from api.database import get_db
from api.models import User
from api.security import enforce_allowed_origin

router = APIRouter(prefix="/me", tags=["explanation-requests"],
    dependencies=[Depends(enforce_allowed_origin)])


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class OpenSession(StrictBody):
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")


class ReleaseSession(StrictBody):
    reason: Literal["leave", "signout"] = "leave"


class ExplainRequest(OpenSession):
    session_id: str = Field(min_length=1, max_length=36)
    operation_id: str = Field(min_length=1, max_length=64)
    timestamp_seconds: float = Field(ge=0, le=172800)
    intervention_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{1,64}$")
    output_mode: Literal["brief", "explanatory", "detailed"] = "explanatory"


class ReviewUpdate(StrictBody):
    heard: bool | None = None
    dismissed: bool | None = None
    partial_seconds: float | None = Field(default=None, ge=0, le=172800)
    satisfaction: Literal["yes", "no"] | None = None


class RetryRequest(StrictBody):
    session_id: str = Field(min_length=1, max_length=36)


async def service(request: Request, db=Depends(get_db)):
    value = getattr(request.app.state, "on_demand_explanations", None)
    if value is None:
        raise HTTPException(503, "On-demand explanations are unavailable.")
    token = value.authorization.set(db.info.get("account_session_digest"))
    try:
        yield value
    finally:
        value.authorization.reset(token)


@router.post("/lecture-sessions", status_code=201)
def open_session(body: OpenSession, user: User = Depends(get_current_user), manager=Depends(service)):
    return manager.open(user.id, body.video_id)


@router.put("/lecture-sessions/{identity}/heartbeat")
def heartbeat(identity: str, user: User = Depends(get_current_user), manager=Depends(service)):
    return manager.heartbeat(user.id, identity)


@router.post("/lecture-sessions/{identity}/release")
def release(identity: str, body: ReleaseSession, user: User = Depends(get_current_user), manager=Depends(service)):
    manager.release(user.id, identity, body.reason)
    return {"status": "released"}


@router.post("/explanation-requests", status_code=202)
def explain(body: ExplainRequest, user: User = Depends(get_current_user), manager=Depends(service)):
    return manager.request(user.id, body.session_id, body.operation_id, body.video_id,
        body.timestamp_seconds, body.output_mode, body.intervention_id)


@router.get("/explanation-requests")
def requests(video_id: str = Query(pattern=r"^[A-Za-z0-9_-]{11}$"), before: str | None = Query(default=None, max_length=36),
    user: User = Depends(get_current_user), manager=Depends(service)):
    return manager.list(user.id, video_id, before)


@router.post("/explanation-requests/{identity}/retry")
def retry(identity: str, body: RetryRequest, user: User = Depends(get_current_user), manager=Depends(service)):
    return manager.retry(user.id, identity, body.session_id)


@router.patch("/explanation-requests/{identity}")
def progress(identity: str, body: ReviewUpdate, user: User = Depends(get_current_user), manager=Depends(service)):
    manager.progress(user.id, identity, body.model_dump(exclude_none=True))
    return manager.get(user.id, identity)
