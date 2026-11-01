from http import HTTPStatus
from typing import Any, cast

import httpx

from ...client import AuthenticatedClient, Client
from ...models.billing_code_summary import BillingCodeSummary
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: BillingCodeSummary,
    opc_request_id: str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(opc_request_id, Unset):
        headers["opc-request-id"] = opc_request_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/billingCodes",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> ErrorResponse | str:
    if response.status_code == 200:
        response_200 = cast(str, response.content)
        return response_200

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
    *,
    client: AuthenticatedClient | Client,
    body: BillingCodeSummary,
    opc_request_id: str | Unset = UNSET,
) -> Response[ErrorResponse | str]:
    """Save list of billing codes to SI.

    Args:
        opc_request_id (str | Unset):
        body (BillingCodeSummary): Summary of billing codes grouped by type for encounterId.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        body=body,
        opc_request_id=opc_request_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: BillingCodeSummary,
    opc_request_id: str | Unset = UNSET,
) -> ErrorResponse | str | None:
    """Save list of billing codes to SI.

    Args:
        opc_request_id (str | Unset):
        body (BillingCodeSummary): Summary of billing codes grouped by type for encounterId.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | str
    """

    return sync_detailed(
        client=client,
        body=body,
        opc_request_id=opc_request_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: BillingCodeSummary,
    opc_request_id: str | Unset = UNSET,
) -> Response[ErrorResponse | str]:
    """Save list of billing codes to SI.

    Args:
        opc_request_id (str | Unset):
        body (BillingCodeSummary): Summary of billing codes grouped by type for encounterId.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        body=body,
        opc_request_id=opc_request_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: BillingCodeSummary,
    opc_request_id: str | Unset = UNSET,
) -> ErrorResponse | str | None:
    """Save list of billing codes to SI.

    Args:
        opc_request_id (str | Unset):
        body (BillingCodeSummary): Summary of billing codes grouped by type for encounterId.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | str
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            opc_request_id=opc_request_id,
        )
    ).parsed
