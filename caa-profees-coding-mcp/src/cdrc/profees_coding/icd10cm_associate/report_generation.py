import pandas as pd

from cdrc.common.report_generation import CSVReportGenerator


class ICD10CMAssociateCSVReportGenerator(CSVReportGenerator):
    """CSV report generator with ICD-10-CM association sample-report ordering."""

    INTERMEDIATE_OUTPUT_COLUMN_ORDER = [
        "output_intermediate_raw_associations",
        "output_intermediate_enhanced_associations",
        "output_intermediate_combo_rule_associations",
        "output_intermediate_validated_associations",
        "output_intermediate_cpt_ncd_associations",
        "output_intermediate_hcpcs_ncd_associations",
        "output_intermediate_regenerated_associations",
        "output_intermediate_regeneration_recommendations",
    ]

    def create_sample_report(self, *args, **kwargs) -> pd.DataFrame:
        report = super().create_sample_report(*args, **kwargs)
        return self._reorder_intermediate_output_columns(report)

    def _reorder_intermediate_output_columns(self, report: pd.DataFrame) -> pd.DataFrame:
        """Keep intermediate output columns aligned with the ICD association stage order."""

        existing_stage_columns = [column for column in self.INTERMEDIATE_OUTPUT_COLUMN_ORDER if column in report.columns]
        if len(existing_stage_columns) <= 1:
            return report

        original_columns = list(report.columns)
        first_stage_index = min(original_columns.index(column) for column in existing_stage_columns)
        remaining_columns = [column for column in original_columns if column not in existing_stage_columns]
        reordered_columns = remaining_columns[:first_stage_index] + existing_stage_columns + remaining_columns[first_stage_index:]

        return report[reordered_columns]
