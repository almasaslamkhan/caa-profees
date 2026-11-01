from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="Complexity")


@_attrs_define
class Complexity:
    """Complexity level for a specific E/M category.

    Attributes:
        level (str | Unset):
        justification (str | Unset):
        confidence (float | Unset):
    """

    level: str | Unset = UNSET
    justification: str | Unset = UNSET
    confidence: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        level = self.level

        justification = self.justification

        confidence = self.confidence

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if level is not UNSET:
            field_dict["level"] = level
        if justification is not UNSET:
            field_dict["justification"] = justification
        if confidence is not UNSET:
            field_dict["confidence"] = confidence

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        level = d.pop("level", UNSET)

        justification = d.pop("justification", UNSET)

        confidence = d.pop("confidence", UNSET)

        complexity = cls(
            level=level,
            justification=justification,
            confidence=confidence,
        )

        complexity.additional_properties = d
        return complexity

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
