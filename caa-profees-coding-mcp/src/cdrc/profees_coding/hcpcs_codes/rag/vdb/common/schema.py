# hcpcs_codes/VDB/common/schema.py
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

GCodeLabel = Literal["YES", "NO"]


class GCodeVDBRecord(BaseModel):
    # identity
    record_id: str = Field(..., description="Unique ID for this G-code justification record")
    example_id: Optional[str] = Field(None, description="Source encounter ID")

    # retrieval core (ONE embedding per record)
    justification_text: str = Field(
        ...,
        description=(
            "GPT-generated justification explaining why the selected "
            "G code applies or does not apply for this encounter. "
            "This text is embedded and stored in the VDB."
        ),
    )

    # supervision (single target G code)
    g_code: str = Field(..., description="G code being justified (e.g., G2211)")
    g_code_label: GCodeLabel = Field(..., description="Whether this G code applies for the encounter")

    # teaching signal (not embedded)
    counterfactual: Optional[str] = Field(
        None,
        description="What would need to change for the G-code decision to flip",
    )

    # audit / versioning
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="prompt_version, model_version, source, etc.",
    )
