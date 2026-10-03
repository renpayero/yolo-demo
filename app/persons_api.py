"""API de identificación: personas registradas (con consentimiento) e interruptor de reconocimiento facial."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.faces import EnrollmentError
from app.storage import Person, PersonNameTaken, Storage

router = APIRouter(prefix="/api", tags=["personas"])


class PersonIn(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=40)]
    consent: bool

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El nombre no puede estar vacío")
        return value


class PersonOut(BaseModel):
    id: int
    name: str
    consent_ts: float
    samples: int

    @classmethod
    def of(cls, person: Person) -> PersonOut:
        return cls(id=person.id, name=person.name, consent_ts=person.consent_ts, samples=person.samples)


class IdentificationIn(BaseModel):
    enabled: bool


def _storage(request: Request) -> Storage:
    return request.app.state.storage


def _capture(request: Request):
    """Toma las muestras de la cara que está frente a la cámara (unos segundos)."""
    try:
        return request.app.state.pipeline.enroll()
    except EnrollmentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _already_registered(request: Request, samples, allowed_id: int | None = None) -> None:
    person_id, _ = request.app.state.pipeline.match_existing(samples)
    if person_id is not None and person_id != allowed_id:
        name = _storage(request).person_names().get(person_id, "otra persona")
        raise HTTPException(status_code=409, detail=f"Esa cara ya está registrada como «{name}».")


@router.get("/persons")
def list_persons(request: Request) -> list[PersonOut]:
    return [PersonOut.of(p) for p in _storage(request).list_persons()]


@router.post("/persons", status_code=201)
def create_person(body: PersonIn, request: Request) -> PersonOut:
    if not body.consent:
        raise HTTPException(status_code=422, detail="Para registrar a una persona hace falta su consentimiento.")
    if body.name in _storage(request).person_names().values():
        raise HTTPException(status_code=409, detail=f"Ya hay una persona registrada como «{body.name}».")
    samples = _capture(request)
    _already_registered(request, samples)
    try:
        person = _storage(request).create_person(body.name, samples)
    except PersonNameTaken:
        raise HTTPException(status_code=409, detail=f"Ya hay una persona registrada como «{body.name}».") from None
    request.app.state.pipeline.reload_persons()
    return PersonOut.of(person)


@router.post("/persons/{person_id}/samples")
def add_samples(person_id: int, request: Request) -> PersonOut:
    """Más muestras de alguien ya registrado (otra luz, otro ángulo): mejora el reconocimiento."""
    if person_id not in _storage(request).person_names():
        raise HTTPException(status_code=404, detail="La persona no existe")
    samples = _capture(request)
    _already_registered(request, samples, allowed_id=person_id)
    _storage(request).add_embeddings(person_id, samples)
    request.app.state.pipeline.reload_persons()
    return next(PersonOut.of(p) for p in _storage(request).list_persons() if p.id == person_id)


@router.delete("/persons/{person_id}")
def delete_person(person_id: int, request: Request) -> dict[str, int]:
    """Borra a la persona y sus huellas faciales; su historial queda, pero anónimo."""
    report = _storage(request).delete_person(person_id)
    if report is None:
        raise HTTPException(status_code=404, detail="La persona no existe")
    request.app.state.pipeline.reload_persons(forget_person_id=person_id)
    return {"visits_anonymized": report.visits_anonymized, "alerts_anonymized": report.alerts_anonymized}


@router.get("/identification")
def get_identification(request: Request) -> dict[str, bool]:
    return {"enabled": request.app.state.pipeline.identification_enabled}


@router.put("/identification")
def set_identification(body: IdentificationIn, request: Request) -> dict[str, bool]:
    request.app.state.pipeline.set_identification(body.enabled)
    return {"enabled": body.enabled}
