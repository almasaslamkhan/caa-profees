from http import HTTPStatus
from typing import Any, cast

import httpx

from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    encounter_id: str,
    *,
    opc_request_id: str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(opc_request_id, Unset):
        headers["opc-request-id"] = opc_request_id

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/v2/billingCodes/{encounter_id}/submit",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> ErrorResponse | str:
    if response.status_code == 201:
        response_201 = cast(str, response.json())
        return response_201

    response_default = ErrorResponse.from_dict(response.json())

    return response_default


def _build_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> Response[ErrorResponse | str]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    encounter_id: str,
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> Response[ErrorResponse | str]:
    """Submit list of billing codes.

    Args:
        encounter_id (str):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        encounter_id=encounter_id,
        opc_request_id=opc_request_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    encounter_id: str,
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> ErrorResponse | str | None:
    """Submit list of billing codes.

    Args:
        encounter_id (str):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | str
    """

    return sync_detailed(
        encounter_id=encounter_id,
        client=client,
        opc_request_id=opc_request_id,
    ).parsed


async def asyncio_detailed(
    encounter_id: str,
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> Response[ErrorResponse | str]:
    """Submit list of billing codes.

    Args:
        encounter_id (str):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        encounter_id=encounter_id,
        opc_request_id=opc_request_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    encounter_id: str,
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> ErrorResponse | str | None:
    """Submit list of billing codes.

    Args:
        encounter_id (str):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | str
    """

    return (
        await asyncio_detailed(
            encounter_id=encounter_id,
            client=client,
            opc_request_id=opc_request_id,
        )
    ).parsed
