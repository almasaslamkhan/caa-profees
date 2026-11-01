from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.codeable_concept_concept_cki import CodeableConceptConceptCki


T = TypeVar("T", bound="CodeableConcept")


@_attrs_define
class CodeableConcept:
    """Concept - reference to a terminology or just text.

    Attributes:
        code (str): Code value in the specified code set.
        text (str | Unset): Textual value of the code used for display.
        coding_key (str | Unset): Code key.
        user_selected (bool | Unset): Indicates if this coding was chosen directly by the user.
        meaning (str | Unset): Meaning of the code.
        concept_cki (CodeableConceptConceptCki | Unset): Cerner key identifier of the code.
    """

    code: str
    text: str | Unset = UNSET
    coding_key: str | Unset = UNSET
    user_selected: bool | Unset = UNSET
    meaning: str | Unset = UNSET
    concept_cki: CodeableConceptConceptCki | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        code = self.code

        text = self.text

        coding_key = self.coding_key

        user_selected = self.user_selected

        meaning = self.meaning

        concept_cki: dict[str, Any] | Unset = UNSET
        if not isinstance(self.concept_cki, Unset):
            concept_cki = self.concept_cki.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "code": code,
            }
        )
        if text is not UNSET:
            field_dict["text"] = text
        if coding_key is not UNSET:
            field_dict["codingKey"] = coding_key
        if user_selected is not UNSET:
            field_dict["userSelected"] = user_selected
        if meaning is not UNSET:
            field_dict["meaning"] = meaning
        if concept_cki is not UNSET:
            field_dict["conceptCki"] = concept_cki

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.codeable_concept_concept_cki import CodeableConceptConceptCki

        d = dict(src_dict)
        code = d.pop("code")

        text = d.pop("text", UNSET)

        coding_key = d.pop("codingKey", UNSET)

        user_selected = d.pop("userSelected", UNSET)

        meaning = d.pop("meaning", UNSET)

        _concept_cki = d.pop("conceptCki", UNSET)
        concept_cki: CodeableConceptConceptCki | Unset
        if isinstance(_concept_cki, Unset):
            concept_cki = UNSET
        else:
            concept_cki = CodeableConceptConceptCki.from_dict(_concept_cki)

        codeable_concept = cls(
            code=code,
            text=text,
            coding_key=coding_key,
            user_selected=user_selected,
            meaning=meaning,
            concept_cki=concept_cki,
        )

        codeable_concept.additional_properties = d
        return codeable_concept

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
