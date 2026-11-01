import logging
from copy import deepcopy

from profees_coding.client.him_coding_service_api_client.models import (
    BillingCodeSummary,
    Complexity,
    Diagnosis,
    Em,
    EmEncounterType,
    EmTimeSpent,
    LinkedDiagnosis,
    Modifier,
    ProcedureCode,
    Reference,
    SourceType,
    Status,
)
from profees_coding.client.him_coding_service_api_client.types import Unset
from profees_coding.common import logging_context

_logger: logging.Logger = logging_context.get_logger(__name__)


class CodingModelMapper:

    def map_billing_code_summary(self, model_response, encounter_id, patient_id, doc_ref_id, encounter_period) -> BillingCodeSummary:
        model_response = self._with_order_catalog_candidates(model_response)
        #  currently there is only one document reference for all the billing codes
        request = BillingCodeSummary(
            encounter_id=encounter_id,
            patient_id=patient_id,
            owner_id=model_response.get("owner_id", patient_id),
            start_date=model_response.get("start_date", ""),
            end_date=model_response.get("start_date", ""),
            procedure=self._map_procedure(model_response.get("procedure", ""), doc_ref_id, encounter_period),
            is_proposed=True,
            em=self._map_EM(model_response.get("em", ""), doc_ref_id),
            diagnosis=self._map_diagnosis(model_response.get("diagnosis", ""), doc_ref_id),
        )
        return request

    def _map_order_catalog_candidates(self, candidates):
        if not isinstance(candidates, list):
            raise ValueError("Order catalog candidates must be a list")
        catalog_ids = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Order catalog candidate must be an object")
            catalog_id = candidate.get("CATALOG_CD")
            if not isinstance(catalog_id, str) or not catalog_id.isascii() or not catalog_id.isdigit() or int(catalog_id) <= 0:
                raise ValueError("CATALOG_CD must be a positive numeric string")
            if catalog_id in catalog_ids:
                raise ValueError(f"Duplicate catalog ID: {catalog_id}")
            catalog_ids.add(catalog_id)
            for field in ("description", "justification"):
                if not isinstance(candidate.get(field), str):
                    raise ValueError(f"Catalog candidate requires {field}")
        return deepcopy(candidates)

    def map_em_order_catalog_groups(self, groups):
        if not isinstance(groups, list):
            raise ValueError("Ambulatory E/M catalog groups must be a list")
        result = deepcopy(groups)
        seen = set()
        for group in result:
            if not isinstance(group, dict) or not all(isinstance(group.get(field), str) and group[field] for field in ("system", "code")):
                raise ValueError("E/M catalog group requires system and code strings")
            key = (group["system"], group["code"])
            if key in seen:
                raise ValueError(f"Duplicate E/M catalog group: {key}")
            seen.add(key)
            group["order_catalog_candidates"] = self._map_order_catalog_candidates(group.get("order_catalog_candidates"))
        return result

    def _with_order_catalog_candidates(self, model_response):
        response = deepcopy(model_response)
        for procedure in response.get("procedure") or []:
            for field in ("quantity", "quantity_evidence"):
                if procedure.get(field) is None:
                    procedure.pop(field, None)
            candidates = procedure.pop("order_catalog_candidates", None)
            if candidates is not None:
                procedure["orderCatalogCandidates"] = self._map_order_catalog_candidates(candidates)

        groups = response.get("ambulatory_em_order_catalog_candidates")
        if groups is not None:
            by_code = {(group["system"], group["code"]): group["order_catalog_candidates"] for group in self.map_em_order_catalog_groups(groups)}
            for em in response.get("em") or []:
                key = (em.get("system", "CPT"), em["code"])
                if key in by_code:
                    em["orderCatalogCandidates"] = by_code[key]
        return response

    def _map_diagnosis(self, codes_from_model, doc_ref_id):
        try:
            # Create a list to store the Diagnosis objects
            diagnosis_objects = []

            # Iterate over the JSON data and create Diagnosis objects
            for item in codes_from_model:
                diagnosis = Diagnosis.from_dict(item)
                diagnosis.type_ = item.get("type_", "DIAGNOSIS")
                if doc_ref_id:
                    # Document reference id is not part of model response, so we have to set manually
                    doc_ref_item = Reference(id=doc_ref_id, type_=SourceType.DOCUMENT_REFERENCE)
                    diagnosis.sources = [doc_ref_item]

                diagnosis_objects.append(diagnosis)
            _logger.info("Successfully loaded diagnosis objects")
            return diagnosis_objects
        except Exception as e:
            _logger.error("Error loading diagnosis : {}".format(e))
            return []

    def _map_procedure(self, codes_from_model, doc_ref_id, encounter_period):
        # Create a list to store the ProcedureCode objects
        procedure_code_objects = []
        try:
            procedure_date = Unset
            if encounter_period and encounter_period["start"]:
                procedure_date = encounter_period["start"]
            # Iterate over the JSON data and create ProcedureCode objects
            for item in codes_from_model:
                procedure_code = ProcedureCode.from_dict(item)
                procedure_code.type_ = item.get("type_", "PROCEDURE")
                procedure_code.procedure_date = procedure_date
                if doc_ref_id:
                    # Document reference id is not part of model response, so we have to set manually
                    doc_ref_item = Reference(id=doc_ref_id, type_=SourceType.DOCUMENT_REFERENCE)
                    procedure_code.sources = [doc_ref_item]
                procedure_code_objects.append(procedure_code)
            _logger.info("Successfully loaded procedure code")
            return procedure_code_objects

        except Exception as e:
            _logger.error("Error loading procedure code: %s", e)
            return []

    def _map_EM(self, codes_from_model, doc_ref_id):
        try:
            em_list = []
            for em_code in codes_from_model:
                em = Em(
                    type_=em_code.get("type", "EM"),
                    description=em_code.get("description", ""),
                    status=Status.PROPOSED,
                    code=em_code.get("code", ""),
                    system=em_code.get("system", ""),
                    justification=em_code.get("justification", ""),
                    source=em_code.get("source", ""),
                    sources=[Reference(id=doc_ref_id, type_=SourceType.DOCUMENT_REFERENCE)] if doc_ref_id else Unset(),
                    time_spent=EmTimeSpent.from_dict(em_code["time_spent"]) if em_code.get("time_spent") else Unset(),
                    modifiers=(
                        [
                            Modifier(
                                code=modifier["code"],
                                description=modifier["description"],
                                justification=modifier["justification"],
                            )
                            for modifier in em_code.get("modifiers")
                        ]
                        if em_code.get("modifiers")
                        else Unset()
                    ),
                    problem_complexity=(
                        Complexity(level=em_code["problem_complexity"]["level"], justification=em_code["problem_complexity"]["justification"])
                        if em_code.get("problem_complexity")
                        else Unset()
                    ),
                    data_complexity=(
                        Complexity(level=em_code["data_complexity"]["level"], justification=em_code["data_complexity"]["justification"])
                        if em_code.get("data_complexity")
                        else Unset()
                    ),
                    risk_of_complications=(
                        Complexity(level=em_code["risk_of_complications"]["level"], justification=em_code["risk_of_complications"]["justification"])
                        if em_code.get("risk_of_complications")
                        else Unset()
                    ),
                    mdm_complexity=(
                        Complexity(level=em_code["mdm_complexity"]["level"], justification=em_code["mdm_complexity"]["justification"])
                        if em_code.get("mdm_complexity")
                        else Unset()
                    ),
                    encounter_type=(
                        EmEncounterType(
                            encounter_type=em_code["encounter_type"]["encounter_type"], justification=em_code["encounter_type"]["justification"]
                        )
                        if em_code.get("encounter_type")
                        else Unset()
                    ),
                    linked_diagnosis=self._map_linked_diagnosis(em_code.get("linked_diagnosis")) if em_code.get("linked_diagnosis") else Unset(),
                )
                if "orderCatalogCandidates" in em_code:
                    em["orderCatalogCandidates"] = em_code["orderCatalogCandidates"]
                em_list.append(em)
            _logger.info("Successfully loaded EM Codes")
            return em_list

        except Exception as e:
            _logger.error(f"While Mapping EM codes ,an error occurred with details : {e}")
            return []

    def _map_linked_diagnosis(self, data):
        list_linked_diagnosis = []
        try:
            for ld_code in data:
                linked_diagnosis = LinkedDiagnosis.from_dict(ld_code)
                list_linked_diagnosis.append(linked_diagnosis)
                _logger.info("Successfully loaded linked diagnosis objects")
        except Exception as e:
            _logger.error("Error loading linked diagnosis : {}".format(e))

        return list_linked_diagnosis


# if __name__ == "__main__":
#     tes = CodingModelMapper()
#     with open("aoa_agent/service/sample.json", "r", encoding="utf-8") as f:
#         raw = json.load(f)
#         # print(raw['em'])
#     print(tes.map_billing_code_summary(raw, "en_abc", "pa_123", "dc-xyz"))
#     # print(tes._map_EM(raw['em']))
