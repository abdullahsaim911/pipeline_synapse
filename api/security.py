"""
Browser-origin controls for the local Synapse API.

This is NOT authentication. The API listens on 127.0.0.1 only, so other
machines cannot reach it, but any web page open in a browser on this computer
can try to call it. These helpers stop that for browsers by allowing only the
Synapse frontend's origin(s):

  * CORS is limited to the allowed origins (no "*"), so other sites cannot read
    responses or pass a preflight.
  * `enforce_allowed_origin` rejects requests that carry a different `Origin`
    header (browsers always send one on cross-origin POSTs), which also stops
    "simple" cross-site requests that CORS alone does not prevent from being sent.

Requests with no `Origin` header (curl, scripts, Electron's main process, other
local programs) are not blocked: any local program can reach the API, as before.

Configuration: SYNAPSE_CORS_ORIGINS, a comma-separated list such as
    http://localhost:5173,http://127.0.0.1:5173
Default: the Vite dev server used by `npm run electron:dev`.

A packaged Electron build loads index.html from a file:// URL, whose requests
carry `Origin: null`. "null" is NOT allowed by default because sandboxed iframes
on any web page also send `Origin: null`; add it to SYNAPSE_CORS_ORIGINS only if
you accept that trade-off.
"""

import logging
import os
import re
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

ORIGINS_ENV_VAR = "SYNAPSE_CORS_ORIGINS"
DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

# scheme://host[:port], no path; or the literal "null".
_ORIGIN_RE = re.compile(r"^(?:null|https?://[A-Za-z0-9.\-]+(?::\d{1,5})?)$")


def get_allowed_origins() -> List[str]:
    """Allowed browser origins from the environment, or the development defaults."""
    raw = os.environ.get(ORIGINS_ENV_VAR)
    if raw is None or not raw.strip():
        return list(DEFAULT_ORIGINS)

    origins: List[str] = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if _ORIGIN_RE.fullmatch(origin):
            if origin not in origins:
                origins.append(origin)
        else:
            logger.warning("Ignoring invalid origin in %s: %r", ORIGINS_ENV_VAR, origin)
    return origins or list(DEFAULT_ORIGINS)


def configure_cors(app: FastAPI) -> None:
    """Add CORS limited to the allowed origins. The frontend sends no cookies."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )


def enforce_allowed_origin(request: Request) -> None:
    """FastAPI dependency: reject browser requests from origins that are not allowed."""
    origin = request.headers.get("origin")
    if origin is not None and origin not in get_allowed_origins():
        raise HTTPException(status_code=403, detail="Origin not allowed")
