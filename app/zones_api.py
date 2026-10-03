"""API REST de zonas: alta, edición, baja, exportación e importación (JSON)."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.storage import Storage, ZoneNameTaken
from app.zones import Zone, ZoneSpec

router = APIRouter(prefix="/api/zones", tags=["zonas"])


Seconds = Annotated[float, Field(ge=1, le=86400)]


class ZoneIn(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=40)]
    kind: Literal["work", "restricted"] = "work"
    polygon: Annotated[list[tuple[float, float]], Field(min_length=3, max_length=64)]
    max_dwell_s: Seconds | None = None
    max_empty_s: Seconds | None = None

    def to_spec(self) -> ZoneSpec:
        # En una restringida cualquier ingreso ya es alerta: los umbrales no aplican.
        work = self.kind == "work"
        return ZoneSpec(
            name=self.name,
            kind=self.kind,
            polygon=tuple((x, y) for x, y in self.polygon),
            max_dwell_s=self.max_dwell_s if work else None,
            max_empty_s=self.max_empty_s if work else None,
        )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El nombre no puede estar vacío")
        return value

    @field_validator("polygon")
    @classmethod
    def _inside_frame(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if any(not 0.0 <= coord <= 1.0 for point in value for coord in point):
            raise ValueError("Las coordenadas tienen que estar normalizadas entre 0 y 1")
        return value


class ZoneOut(BaseModel):
    id: int
    name: str
    kind: str
    polygon: list[tuple[float, float]]
    color: str
    max_dwell_s: float | None
    max_empty_s: float | None

    @classmethod
    def of(cls, zone: Zone) -> ZoneOut:
        return cls(
            id=zone.id, name=zone.name, kind=zone.kind, polygon=list(zone.polygon), color=zone.color,
            max_dwell_s=zone.max_dwell_s, max_empty_s=zone.max_empty_s,
        )


class ZonesFile(BaseModel):
    version: Literal[1] = 1
    camera_id: str | None = None
    zones: list[ZoneIn]


def _storage(request: Request) -> Storage:
    return request.app.state.storage


def _camera_id(request: Request) -> str:
    return request.app.state.cfg.camera.id


def _zones_changed(request: Request) -> None:
    request.app.state.pipeline.request_zones_reload()


def _name_taken(name: str) -> HTTPException:
    return HTTPException(status_code=409, detail=f"Ya existe una zona llamada «{name}»")


@router.get("")
def list_zones(request: Request) -> list[ZoneOut]:
    return [ZoneOut.of(z) for z in _storage(request).list_zones(_camera_id(request))]


@router.post("", status_code=201)
def create_zone(body: ZoneIn, request: Request) -> ZoneOut:
    try:
        zone = _storage(request).create_zone(_camera_id(request), body.to_spec())
    except ZoneNameTaken:
        raise _name_taken(body.name) from None
    _zones_changed(request)
    return ZoneOut.of(zone)


@router.put("/{zone_id}")
def update_zone(zone_id: int, body: ZoneIn, request: Request) -> ZoneOut:
    try:
        zone = _storage(request).update_zone(zone_id, body.to_spec())
    except ZoneNameTaken:
        raise _name_taken(body.name) from None
    if zone is None:
        raise HTTPException(status_code=404, detail="La zona no existe")
    _zones_changed(request)
    return ZoneOut.of(zone)


@router.delete("/{zone_id}", status_code=204)
def delete_zone(zone_id: int, request: Request) -> Response:
    if not _storage(request).delete_zone(zone_id):
        raise HTTPException(status_code=404, detail="La zona no existe")
    _zones_changed(request)
    return Response(status_code=204)


@router.get("/export")
def export_zones(request: Request) -> JSONResponse:
    camera_id = _camera_id(request)
    zones = _storage(request).list_zones(camera_id)
    payload = ZonesFile(
        camera_id=camera_id,
        zones=[
            ZoneIn(
                name=z.name, kind=z.kind, polygon=list(z.polygon),  # type: ignore[arg-type]
                max_dwell_s=z.max_dwell_s, max_empty_s=z.max_empty_s,
            )
            for z in zones
        ],
    )
    return JSONResponse(
        payload.model_dump(),
        headers={"Content-Disposition": f'attachment; filename="zonas-{camera_id}.json"'},
    )


@router.post("/import")
def import_zones(body: ZonesFile, request: Request) -> list[ZoneOut]:
    """Reemplaza todas las zonas de la cámara por las del archivo."""
    try:
        zones = _storage(request).replace_zones(_camera_id(request), [z.to_spec() for z in body.zones])
    except ZoneNameTaken:
        raise HTTPException(status_code=409, detail="El archivo tiene zonas con nombres repetidos") from None
    _zones_changed(request)
    return [ZoneOut.of(z) for z in zones]
