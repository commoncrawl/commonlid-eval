"""FastAPI layer over :class:`~commonlid.server.service.ScoringService`.

Endpoints. When ``api_token`` is configured, every endpoint except
``/health`` needs ``Authorization: Bearer <token>``:

* ``GET  /health`` — liveness probe.
* ``GET  /status`` — datasets, returned metrics and the submission budget.
* ``GET  /datasets/{dataset_id}/inputs`` — label-free ``[{id, text}]`` items,
  paginated with ``offset`` / ``limit``.
* ``POST /datasets/{dataset_id}/submissions`` — score predictions.
* ``GET  /submissions`` — past submissions and their returned scores.

Interactive API docs are served at ``/docs``.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from commonlid import __version__
from commonlid.server.service import ScoringError, ScoringService

# Module level: with postponed annotations FastAPI resolves the dependency in
# `_authorize`'s signature against this module's globals.
_bearer = HTTPBearer(auto_error=False)


class Prediction(BaseModel):
    id: int = Field(description="Item id as returned by the inputs endpoint.")
    pred: str | None = Field(
        description="Predicted language code (ISO 639-3 preferred; ISO 639-1/2 are "
        "upgraded). null counts as an 'und' prediction."
    )


class Submission(BaseModel):
    predictions: list[Prediction] = Field(
        description="One prediction for every item id of the dataset."
    )


def create_app(service: ScoringService) -> FastAPI:
    """Build the HTTP app around an already-initialised service."""
    app = FastAPI(
        title="CommonLID scoring server",
        version=__version__,
        description=(
            "Fetch label-free inputs, submit language predictions and get scores back. "
            "Each accepted submission counts against a fixed budget; malformed or "
            "over-budget submissions are rejected and not counted."
        ),
    )

    token = service.config.api_token

    def _authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> None:
        if token is None:
            return
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode("utf-8"), token.encode("utf-8")
        ):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    authorized = [Depends(_authorize)]

    @app.exception_handler(ScoringError)
    def _scoring_error(_: Request, exc: ScoringError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status", dependencies=authorized)
    def status() -> dict[str, Any]:
        return service.status()

    @app.get("/datasets/{dataset_id}/inputs", dependencies=authorized)
    def inputs(
        dataset_id: str,
        offset: int = Query(0, ge=0),
        limit: int | None = Query(None, ge=1, description="Page size; omit for all items."),
    ) -> dict[str, Any]:
        return service.inputs(dataset_id, offset=offset, limit=limit)

    @app.post("/datasets/{dataset_id}/submissions", dependencies=authorized)
    def submit(dataset_id: str, submission: Submission) -> dict[str, Any]:
        return service.submit(dataset_id, ((p.id, p.pred) for p in submission.predictions))

    @app.get("/submissions", dependencies=authorized)
    def submissions() -> list[dict[str, Any]]:
        return service.history()

    return app
