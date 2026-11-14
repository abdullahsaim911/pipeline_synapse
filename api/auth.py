"""Local account identity. Shared pipeline work never depends on browser sessions."""
from datetime import datetime, timedelta, timezone
import hashlib
import os
import secrets
import threading
import time
from collections import OrderedDict

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import event, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import AccountCredential, AccountSession, LectureSessionInterest, User, LOCAL_DEVELOPMENT_USER_ID, LOCAL_DEVELOPMENT_PROFILE_KEY
from api.security import enforce_allowed_origin

LOCAL_ENVIRONMENTS = {"", "dev", "development", "local", "test"}
SESSION_DAYS = 30
passwords = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=1)
_hash_slots = threading.BoundedSemaphore(2)
_attempt_lock = threading.Lock()
_attempts = OrderedDict()
_dummy_hash = None


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def local_transport(request: Request):
    if os.getenv("SYNAPSE_ENV", "development").strip().lower() not in LOCAL_ENVIRONMENTS:
        raise HTTPException(404, "Local accounts are unavailable in this environment.")
    if not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "Accounts are available only on this computer.")


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def bearer(request):
    raw = request.headers.get("authorization")
    if raw is None:
        return None
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer" or not 32 <= len(token) <= 256 or not token.isascii() or any(c.isspace() for c in token):
        raise HTTPException(401, "Invalid or expired session.")
    return token


def get_current_user(request: Request, response: Response, db: Session = Depends(get_db)) -> User:
    response.headers["Cache-Control"] = "private, no-store"
    enforce_allowed_origin(request)
    local_transport(request)
    token = bearer(request)
    if token is None:
        if os.getenv("SYNAPSE_LOCAL_PROFILE_COMPAT", "0").strip().lower() not in {"1", "true"}:
            raise HTTPException(401, "Sign in required.")
        user = db.query(User).filter_by(id=LOCAL_DEVELOPMENT_USER_ID,
            profile_key=LOCAL_DEVELOPMENT_PROFILE_KEY, is_local=True).one_or_none()
        if user is None:
            raise HTTPException(503, "Local development profile is unavailable.")
        return user
    row = db.query(AccountSession).filter(AccountSession.token_digest == digest(token),
        AccountSession.revoked_at.is_(None), AccountSession.expires_at > utcnow()).one_or_none()
    if row is None:
        raise HTTPException(401, "Invalid or expired session.")
    user = db.get(User, row.user_id)
    if user is None or user.is_local:
        raise HTTPException(401, "Invalid or expired session.")
    db.info["account_session_digest"] = row.token_digest
    request.state.account_session = row
    return user


@event.listens_for(Session, "before_commit")
def check_session_at_commit(db):
    """Serialize a personal commit with revocation, including slow uploads.

    The conditional write takes the database's write lock. Logout before this
    point rejects the commit; logout after it waits for this commit to finish.
    """
    key = db.info.get("account_session_digest")
    if key:
        result = db.execute(update(AccountSession).where(AccountSession.token_digest == key,
            AccountSession.revoked_at.is_(None), AccountSession.expires_at > utcnow()).values(
                revoked_at=AccountSession.revoked_at).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            raise HTTPException(401, "Invalid or expired session.")


class BoundedAccountRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        async def bounded(request):
            enforce_allowed_origin(request)
            local_transport(request)
            if request.method == "POST":
                body = bytearray()
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 8192:
                        raise HTTPException(413, "Account request is too large.")
                    body.extend(chunk)
                request._body = bytes(body)
            return await original(request)
        return bounded


router = APIRouter(prefix="/auth", tags=["local-accounts"], route_class=BoundedAccountRoute,
    dependencies=[Depends(enforce_allowed_origin), Depends(local_transport)])


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    password: str = Field(min_length=8, max_length=128)


class Signup(Credentials):
    display_name: str = Field(default="", max_length=120)


def throttle(request, username):
    now = time.monotonic()
    # One client bucket also bounds attempts using a different username each time.
    keys = ("client:" + request.client.host, "user:" + username.lower())
    with _attempt_lock:
        for key in keys:
            previous = _attempts.get(key, [])
            previous = [stamp for stamp in previous if now - stamp < 60]
            if len(previous) >= (30 if key.startswith("client:") else 10):
                raise HTTPException(429, "Too many attempts. Try again in a minute.", headers={"Retry-After": "60"})
            _attempts[key] = previous
        for key in keys:
            _attempts[key].append(now)
            _attempts.move_to_end(key)
        while len(_attempts) > 2048:
            _attempts.popitem(last=False)


def user_response(user):
    return {"id": user.id, "displayName": user.display_name, "isLocalProfile": user.is_local}


def new_session(db, user):
    token = secrets.token_urlsafe(48)
    now = utcnow()
    expiry = now + timedelta(days=SESSION_DAYS)
    db.add(AccountSession(token_digest=digest(token), user_id=user.id, created_at=now, expires_at=expiry))
    db.commit()
    return {"user": user_response(user), "token": token, "expiresAt": expiry.isoformat() + "Z"}


@router.post("/signup", status_code=201)
def signup(body: Signup, request: Request, response: Response, db: Session = Depends(get_db)):
    response.headers["Cache-Control"] = "no-store"
    username = body.username.lower()
    throttle(request, username)
    if not _hash_slots.acquire(blocking=False):
        raise HTTPException(429, "Account service is busy. Try again shortly.")
    try:
        hashed = passwords.hash(body.password)
    finally:
        _hash_slots.release()
    user = User(profile_key="account:" + secrets.token_hex(20), display_name=body.display_name.strip() or body.username, is_local=False)
    try:
        db.add(user)
        db.flush()
        db.add(AccountCredential(user_id=user.id, username=username, password_hash=hashed))
        db.flush()
        return new_session(db, user)
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(409, "Username is unavailable.") from error


@router.post("/login")
def login(body: Credentials, request: Request, response: Response, db: Session = Depends(get_db)):
    global _dummy_hash
    response.headers["Cache-Control"] = "no-store"
    throttle(request, body.username)
    row = db.query(AccountCredential).filter_by(username=body.username.lower()).one_or_none()
    if not _hash_slots.acquire(blocking=False):
        raise HTTPException(429, "Account service is busy. Try again shortly.")
    try:
        if _dummy_hash is None:
            _dummy_hash = passwords.hash(secrets.token_urlsafe(32))
        try:
            valid = passwords.verify(row.password_hash if row else _dummy_hash, body.password)
        except (VerificationError, InvalidHashError):
            valid = False
        if not row or not valid:
            raise HTTPException(401, "Invalid username or password.")
        if passwords.check_needs_rehash(row.password_hash):
            row.password_hash = passwords.hash(body.password)
    finally:
        _hash_slots.release()
    return new_session(db, db.get(User, row.user_id))


@router.get("/session")
def session(request: Request, response: Response, user: User = Depends(get_current_user)):
    response.headers["Cache-Control"] = "no-store"
    active = getattr(request.state, "account_session", None)
    return {"user": user_response(user), "expiresAt": active.expires_at.isoformat() + "Z" if active else None}


@router.post("/logout", status_code=204)
def logout(request: Request, db: Session = Depends(get_db)):
    token = bearer(request)
    if token:
        db.execute(update(LectureSessionInterest).where(LectureSessionInterest.account_digest == digest(token),
            LectureSessionInterest.state == "active").values(state="continuing", updated_at=utcnow()))
        db.execute(update(AccountSession).where(AccountSession.token_digest == digest(token),
            AccountSession.revoked_at.is_(None)).values(revoked_at=utcnow()))
        db.commit()
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
