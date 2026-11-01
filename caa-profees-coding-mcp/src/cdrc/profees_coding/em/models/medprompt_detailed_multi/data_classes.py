from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSION = "prompt_version"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    EMBEDDING_MODEL = "embedding_model"
    VDB_INDEX_PATH = "vdb_index_path"
    NUM_FEW_SHOT_EXAMPLES = "num_few_shot_examples"
    NUM_REPEATS = "num_repeats"
    PREDICT_MDM_COMPLEXITY = "predict_mdm_complexity"
    USE_INTERMEDIATE_COMPLEXITIES = "use_intermediate_complexities"
    RCI_MODE = "rci_mode"
    RCI_PROMPT_VERSION_PC = "rci_prompt_version_pc"
    RCI_PROMPT_VERSION_DC = "rci_prompt_version_dc"
    RCI_PROMPT_VERSION_RC = "rci_prompt_version_rc"
    ENCOUNTER_TYPE_PROMPT_VERSION = "encounter_type_prompt_version"
    OMIT_SOAP_NOTE = "omit_soap_note"
    OMIT_PATIENT_PROBLEMS = "omit_patient_problems"
    ICL_MODE = "icl_mode"
    RCI_RC_MODE = "rci_rc_mode"
    RCI_DC_MODE = "rci_dc_mode"
    RCI_PC_MODE = "rci_pc_mode"
    TIME_STATEMENT_PROMPT_VERSION = "time_statement_prompt_version"
    ENCOUNTER_PROMPT_KEY = "encounter_prompt_key"
    TIME_STATEMENT_PROMPT_KEY = "time_statement_prompt_key"
    MDM_PROMPT_KEY = "mdm_prompt_key"
    RCI_MULTI_CRITIC_PROMPT_KEY = "rci_multi_critic_prompt_key"
    PATIENT_TYPE_PROMPT_VERSION = "patient_type_prompt_version"


class PROMPTS(StringEnum):
    TIME_STATEMENT = "time_statement"
    ENCOUNTER = "encounter_type"
    ENCOUNTER_ED = "encounter_type_ed"
    ENCOUNTER_INPATIENT = "encounter_type_inpatient"
    TIME_STATEMENT_INPATIENT = "time_statement_inpatient"
    PATIENT_TYPE = "patient_type"
    MDM = "medprompt_inference"
    MDM_ED = "medprompt_inference_ed"
    MDM_INPATIENT = "medprompt_inference_inpatient"
    MEDPROMPT_OURS = "medprompt_cot_extraction_ours"
    RCI_CRITIC = "rci_critic"
    RCI_IMPROVE = "rci_improve"
    RCI_MULTI_CRITIC = "rci_multi_critic"
    RCI_MULTI_CRITIC_ED = "rci_multi_critic_ed"
    RCI_MULTI_CRITIC_INPATIENT = "rci_multi_critic_inpatient"


class METADATA_KEYS(StringEnum):
    SOAP_NOTE = "soap_note"
    CHAIN_OF_THOUGHT_REASONING = "chain_of_thought_reasoning"
