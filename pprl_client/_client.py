from json import JSONDecodeError

import httpx
from pprl_model import (
    VectorMatchRequest,
    VectorMatchResponse,
    EntityTransformRequest,
    EntityTransformResponse,
    EntityMaskRequest,
    EntityMaskResponse,
)
from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar

_MI = TypeVar("_MI", bound=BaseModel)
_MO = TypeVar("_MO", bound=BaseModel)


class GenericErrorResponse(BaseModel):
    detail: str


class ValidationErrorDetail(BaseModel):
    loc: list[str]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorDetail]


class PPRLError(httpx.HTTPError):
    def __init__(
        self, message: str, request: httpx.Request, error: GenericErrorResponse | ValidationErrorResponse = None
    ):
        super().__init__(message)
        self._request = request
        self.error_response = error

        self.error_type = "unknown"

        if isinstance(error, GenericErrorResponse):
            self.error_type = "default"

        if isinstance(error, ValidationErrorResponse):
            self.error_type = "validation"


def new_error_from_response(r: httpx.Response):
    error_response = None
    error_message = f"received status code {r.status_code}"

    # validation error (422 by default with FastAPI)
    if r.status_code == httpx.codes.UNPROCESSABLE_ENTITY.value:
        try:
            error_response = ValidationErrorResponse(**r.json())
            error_message += ": invalid request"
        except (ValidationError, JSONDecodeError):
            pass
    else:
        try:
            error_response = GenericErrorResponse(**r.json())
            error_message += f": {error_response.detail}"
        except (ValidationError, JSONDecodeError):
            pass

    return PPRLError(error_message, r.request, error_response)


class PPRLClient(object):
    def __init__(self, client: httpx.Client = None, base_url: str = None):
        self._client = client or httpx.Client(base_url=base_url)

    def _request(self, path: str, model_in: _MI, model_out: type[_MO]) -> _MO:
        r = self._client.post(path, json=model_in.model_dump(mode="json"))

        # we generally expect a 200 here
        if r.status_code != httpx.codes.OK.value:
            raise new_error_from_response(r)

        return model_out(**r.json())

    def match(self, request: VectorMatchRequest):
        return self._request("match/", request, VectorMatchResponse)

    def transform(self, request: EntityTransformRequest):
        return self._request("transform/", request, EntityTransformResponse)

    def mask(self, request: EntityMaskRequest):
        return self._request("mask/", request, EntityMaskResponse)
