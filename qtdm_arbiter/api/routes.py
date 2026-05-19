from __future__ import annotations

from fastapi import APIRouter

from qtdm_arbiter.arbiter_decide import run_request
from qtdm_arbiter.models.request import ArbiterRequest
from qtdm_arbiter.models.response import ArbiterResponse


router = APIRouter()


@router.post("/arbiter/decide", response_model=ArbiterResponse)
def arbiter_decide(request: ArbiterRequest) -> ArbiterResponse:
    return run_request(request)
