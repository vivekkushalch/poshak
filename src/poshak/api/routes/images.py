"""Serve stored photos. Unauthenticated, like any public image host."""

import mimetypes

from fastapi import APIRouter, Response

from ..deps import ContainerDep

router = APIRouter(prefix="/img", tags=["images"])


def _image(data: bytes, name: str) -> Response:
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return Response(data, media_type=media_type, headers={"Cache-Control": "max-age=3600"})


@router.get("/original/{name}")
def original(c: ContainerDep, name: str) -> Response:
    return _image(c.originals.get(name), name)


@router.get("/rmbg/{name}")
def cutout(c: ContainerDep, name: str) -> Response:
    return _image(c.cutouts.get(name), name)
