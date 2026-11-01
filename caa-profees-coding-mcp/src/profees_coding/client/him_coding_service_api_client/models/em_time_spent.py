from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="EmTimeSpent")


@_attrs_define
class EmTimeSpent:
    """
    Attributes:
        time_spent_minutes (int | Unset):  Example: 14.
        evidence (str | Unset):
    """

    time_spent_minutes: int | Unset = UNSET
    evidence: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        time_spent_minutes = self.time_spent_minutes

        evidence = self.evidence

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if time_spent_minutes is not UNSET:
            field_dict["time_spent_minutes"] = time_spent_minutes
        if evidence is not UNSET:
            field_dict["evidence"] = evidence

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        time_spent_minutes = d.pop("time_spent_minutes", UNSET)

        evidence = d.pop("evidence", UNSET)

        em_time_spent = cls(
            time_spent_minutes=time_spent_minutes,
            evidence=evidence,
        )

        em_time_spent.additional_properties = d
        return em_time_spent

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
