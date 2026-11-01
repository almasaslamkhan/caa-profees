from http import HTTPStatus
from typing import Any

import httpx

from ...client import AuthenticatedClient, Client
from ...models.billing_code_summary import BillingCodeSummary
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    encounter_id: str,
    status: str | Unset = UNSET,
    type_: str | Unset = UNSET,
    opc_request_id: str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(opc_request_id, Unset):
        headers["opc-request-id"] = opc_request_id

    params: dict[str, Any] = {}

    params["encounterId"] = encounter_id

    params["status"] = status

    params["type"] = type_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/billingCodes",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> BillingCodeSummary | ErrorResponse:
    if response.status_code == 200:
        response_200 = BillingCodeSummary.from_dict(response.json())

        return response_200

    response_default = ErrorResponse.from_dict(response.json())

    return response_default


def _build_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> Response[BillingCodeSummary | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    encounter_id: str,
    status: str | Unset = UNSET,
    type_: str | Unset = UNSET,
    opc_request_id: str | Unset = UNSET,
) -> Response[BillingCodeSummary | ErrorResponse]:
    """Return list of billing codes from SI.

    Args:
        encounter_id (str):
        status (str | Unset):
        type_ (str | Unset):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BillingCodeSummary | ErrorResponse]
    """

    kwargs = _get_kwargs(
        encounter_id=encounter_id,
        status=status,
        type_=type_,
        opc_request_id=opc_request_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    encounter_id: str,
    status: str | Unset = UNSET,
    type_: str | Unset = UNSET,
    opc_request_id: str | Unset = UNSET,
) -> BillingCodeSummary | ErrorResponse | None:
    """Return list of billing codes from SI.

    Args:
        encounter_id (str):
        status (str | Unset):
        type_ (str | Unset):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BillingCodeSummary | ErrorResponse
    """

    return sync_detailed(
        client=client,
        encounter_id=encounter_id,
        status=status,
        type_=type_,
        opc_request_id=opc_request_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    encounter_id: str,
    status: str | Unset = UNSET,
    type_: str | Unset = UNSET,
    opc_request_id: str | Unset = UNSET,
) -> Response[BillingCodeSummary | ErrorResponse]:
    """Return list of billing codes from SI.

    Args:
        encounter_id (str):
        status (str | Unset):
        type_ (str | Unset):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BillingCodeSummary | ErrorResponse]
    """

    kwargs = _get_kwargs(
        encounter_id=encounter_id,
        status=status,
        type_=type_,
        opc_request_id=opc_request_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    encounter_id: str,
    status: str | Unset = UNSET,
    type_: str | Unset = UNSET,
    opc_request_id: str | Unset = UNSET,
) -> BillingCodeSummary | ErrorResponse | None:
    """Return list of billing codes from SI.

    Args:
        encounter_id (str):
        status (str | Unset):
        type_ (str | Unset):
        opc_request_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BillingCodeSummary | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            encounter_id=encounter_id,
            status=status,
            type_=type_,
            opc_request_id=opc_request_id,
        )
    ).parsed
