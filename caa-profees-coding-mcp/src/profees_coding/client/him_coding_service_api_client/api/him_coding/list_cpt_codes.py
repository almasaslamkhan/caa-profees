from http import HTTPStatus
from typing import Any

import httpx

from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.list_cpt_codes_response_200 import ListCptCodesResponse200
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    opc_request_id: str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(opc_request_id, Unset):
        headers["opc-request-id"] = opc_request_id

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/cptCodes",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> ErrorResponse | ListCptCodesResponse200:
    if response.status_code == 200:
        response_200 = ListCptCodesResponse200.from_dict(response.json())

        return response_200

    response_default = ErrorResponse.from_dict(response.json())

    return response_default


def _build_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> Response[ErrorResponse | ListCptCodesResponse200]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> Response[ErrorResponse | ListCptCodesResponse200]:
    """Return list of CPT codes.

    Args:
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ListCptCodesResponse200]
    """

    kwargs = _get_kwargs(
        opc_request_id=opc_request_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> ErrorResponse | ListCptCodesResponse200 | None:
    """Return list of CPT codes.

    Args:
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ListCptCodesResponse200
    """

    return sync_detailed(
        client=client,
        opc_request_id=opc_request_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> Response[ErrorResponse | ListCptCodesResponse200]:
    """Return list of CPT codes.

    Args:
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ListCptCodesResponse200]
    """

    kwargs = _get_kwargs(
        opc_request_id=opc_request_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    opc_request_id: str | Unset = UNSET,
) -> ErrorResponse | ListCptCodesResponse200 | None:
    """Return list of CPT codes.

    Args:
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ListCptCodesResponse200
    """

    return (
        await asyncio_detailed(
            client=client,
            opc_request_id=opc_request_id,
        )
    ).parsed
