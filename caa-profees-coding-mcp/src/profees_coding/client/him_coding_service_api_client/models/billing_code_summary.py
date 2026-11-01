from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.diagnosis import Diagnosis
    from ..models.em import Em
    from ..models.grouper import Grouper
    from ..models.procedure_code import ProcedureCode


T = TypeVar("T", bound="BillingCodeSummary")


@_attrs_define
class BillingCodeSummary:
    """Summary of billing codes grouped by type for encounterId.

    Attributes:
        encounter_id (str | Unset):
        patient_id (str | Unset):
        owner_id (str | Unset):
        start_date (str | Unset):
        end_date (str | Unset):
        is_proposed (bool | Unset):
        procedure (list[ProcedureCode] | Unset):
        em (list[Em] | Unset):
        diagnosis (list[Diagnosis] | Unset):
        grouper (list[Grouper] | Unset):
    """

    encounter_id: str | Unset = UNSET
    patient_id: str | Unset = UNSET
    owner_id: str | Unset = UNSET
    start_date: str | Unset = UNSET
    end_date: str | Unset = UNSET
    is_proposed: bool | Unset = UNSET
    procedure: list[ProcedureCode] | Unset = UNSET
    em: list[Em] | Unset = UNSET
    diagnosis: list[Diagnosis] | Unset = UNSET
    grouper: list[Grouper] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        encounter_id = self.encounter_id

        patient_id = self.patient_id

        owner_id = self.owner_id

        start_date = self.start_date

        end_date = self.end_date

        is_proposed = self.is_proposed

        procedure: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.procedure, Unset):
            procedure = []
            for procedure_item_data in self.procedure:
                procedure_item = procedure_item_data.to_dict()
                procedure.append(procedure_item)

        em: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.em, Unset):
            em = []
            for em_item_data in self.em:
                em_item = em_item_data.to_dict()
                em.append(em_item)

        diagnosis: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.diagnosis, Unset):
            diagnosis = []
            for diagnosis_item_data in self.diagnosis:
                diagnosis_item = diagnosis_item_data.to_dict()
                diagnosis.append(diagnosis_item)

        grouper: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.grouper, Unset):
            grouper = []
            for grouper_item_data in self.grouper:
                grouper_item = grouper_item_data.to_dict()
                grouper.append(grouper_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if encounter_id is not UNSET:
            field_dict["encounterId"] = encounter_id
        if patient_id is not UNSET:
            field_dict["patientId"] = patient_id
        if owner_id is not UNSET:
            field_dict["ownerId"] = owner_id
        if start_date is not UNSET:
            field_dict["start_date"] = start_date
        if end_date is not UNSET:
            field_dict["end_date"] = end_date
        if is_proposed is not UNSET:
            field_dict["isProposed"] = is_proposed
        if procedure is not UNSET:
            field_dict["procedure"] = procedure
        if em is not UNSET:
            field_dict["em"] = em
        if diagnosis is not UNSET:
            field_dict["diagnosis"] = diagnosis
        if grouper is not UNSET:
            field_dict["grouper"] = grouper

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.diagnosis import Diagnosis
        from ..models.em import Em
        from ..models.grouper import Grouper
        from ..models.procedure_code import ProcedureCode

        d = dict(src_dict)
        encounter_id = d.pop("encounterId", UNSET)

        patient_id = d.pop("patientId", UNSET)

        owner_id = d.pop("ownerId", UNSET)

        start_date = d.pop("start_date", UNSET)

        end_date = d.pop("end_date", UNSET)

        is_proposed = d.pop("isProposed", UNSET)

        _procedure = d.pop("procedure", UNSET)
        procedure: list[ProcedureCode] | Unset = UNSET
        if _procedure is not UNSET:
            procedure = []
            for procedure_item_data in _procedure:
                procedure_item = ProcedureCode.from_dict(procedure_item_data)

                procedure.append(procedure_item)

        _em = d.pop("em", UNSET)
        em: list[Em] | Unset = UNSET
        if _em is not UNSET:
            em = []
            for em_item_data in _em:
                em_item = Em.from_dict(em_item_data)

                em.append(em_item)

        _diagnosis = d.pop("diagnosis", UNSET)
        diagnosis: list[Diagnosis] | Unset = UNSET
        if _diagnosis is not UNSET:
            diagnosis = []
            for diagnosis_item_data in _diagnosis:
                diagnosis_item = Diagnosis.from_dict(diagnosis_item_data)

                diagnosis.append(diagnosis_item)

        _grouper = d.pop("grouper", UNSET)
        grouper: list[Grouper] | Unset = UNSET
        if _grouper is not UNSET:
            grouper = []
            for grouper_item_data in _grouper:
                grouper_item = Grouper.from_dict(grouper_item_data)

                grouper.append(grouper_item)

        billing_code_summary = cls(
            encounter_id=encounter_id,
            patient_id=patient_id,
            owner_id=owner_id,
            start_date=start_date,
            end_date=end_date,
            is_proposed=is_proposed,
            procedure=procedure,
            em=em,
            diagnosis=diagnosis,
            grouper=grouper,
        )

        billing_code_summary.additional_properties = d
        return billing_code_summary

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
