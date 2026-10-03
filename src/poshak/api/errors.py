"""Domain errors -> HTTP responses, in one place. Services never raise HTTPException."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..domain.errors import (
    IndexNotReady,
    InvalidImage,
    InvalidKey,
    NotFound,
    Overloaded,
    PoshakError,
)

_STATUS: dict[type[PoshakError], int] = {
    InvalidImage: 400,
    InvalidKey: 400,
    NotFound: 404,
    IndexNotReady: 503,
    Overloaded: 503,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PoshakError)
    async def _domain_error(request: Request, exc: PoshakError) -> JSONResponse:
        status = next((code for t, code in _STATUS.items() if isinstance(exc, t)), 500)
        headers = {"Retry-After": "1"} if isinstance(exc, Overloaded) else None
        return JSONResponse({"detail": str(exc)}, status_code=status, headers=headers)
