from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.status import Status
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.reference import Reference
    from ..models.sub_type import SubType


T = TypeVar("T", bound="BillingCode")


@_attrs_define
class BillingCode:
    """The billing code details common for all types.

    Attributes:
        type_ (str | Unset): Classification tag. Common values include DIAGNOSIS, EM, PROCEDURE, PREVENTIVE. For
            program-specific classifications, this may also be "G-code" (e.g., under diagnosis) or "Z-code" (e.g., under
            preventive), depending on workflow needs.
        code (str | Unset):
        description (str | Unset):
        system (str | Unset):
        justification (str | Unset):
        status (Status | Unset):
        sub_type (SubType | Unset): Identifier for proposed or accepted code.
        sources (list[Reference] | Unset):
    """

    type_: str | Unset = UNSET
    code: str | Unset = UNSET
    description: str | Unset = UNSET
    system: str | Unset = UNSET
    justification: str | Unset = UNSET
    status: Status | Unset = UNSET
    sub_type: SubType | Unset = UNSET
    sources: list[Reference] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        code = self.code

        description = self.description

        system = self.system

        justification = self.justification

        status: str | Unset = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value

        sub_type: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sub_type, Unset):
            sub_type = self.sub_type.to_dict()

        sources: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.sources, Unset):
            sources = []
            for sources_item_data in self.sources:
                sources_item = sources_item_data.to_dict()
                sources.append(sources_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_
        if code is not UNSET:
            field_dict["code"] = code
        if description is not UNSET:
            field_dict["description"] = description
        if system is not UNSET:
            field_dict["system"] = system
        if justification is not UNSET:
            field_dict["justification"] = justification
        if status is not UNSET:
            field_dict["status"] = status
        if sub_type is not UNSET:
            field_dict["sub_type"] = sub_type
        if sources is not UNSET:
            field_dict["sources"] = sources

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.reference import Reference
        from ..models.sub_type import SubType

        d = dict(src_dict)
        type_ = d.pop("type", UNSET)

        code = d.pop("code", UNSET)

        description = d.pop("description", UNSET)

        system = d.pop("system", UNSET)

        justification = d.pop("justification", UNSET)

        _status = d.pop("status", UNSET)
        status: Status | Unset
        if isinstance(_status, Unset):
            status = UNSET
        else:
            status = Status(_status)

        _sub_type = d.pop("sub_type", UNSET)
        sub_type: SubType | Unset
        if isinstance(_sub_type, Unset):
            sub_type = UNSET
        else:
            sub_type = SubType.from_dict(_sub_type)

        _sources = d.pop("sources", UNSET)
        sources: list[Reference] | Unset = UNSET
        if _sources is not UNSET:
            sources = []
            for sources_item_data in _sources:
                sources_item = Reference.from_dict(sources_item_data)

                sources.append(sources_item)

        billing_code = cls(
            type_=type_,
            code=code,
            description=description,
            system=system,
            justification=justification,
            status=status,
            sub_type=sub_type,
            sources=sources,
        )

        billing_code.additional_properties = d
        return billing_code

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
