# Medical Current Procedural Terminology (CPT) Coding Project

This project is designed to automate the process of assigning CPT codes to medical encounters, particularly focusing on Evaluation and Management (E/M) services. It includes various components for data processing, model implementation, and evaluation.


## Key Components
Please follow the below conventions for the project structure. Each subproject (e.g. CPT EM Coding `em`) comprises of the following directories:
- `data/`: Contains schema definitions and data reading utilities.
- `dataset/`: Includes datasets for testing and development.
- `models/`: Contains various model implementations in langgraph, each under a separate subdirectory. Current models include:
  - Charge assist model flow (`/models/charge_assist/`),
  - LLM only method (`/models/llm_only/`), and
  - Decision tree graph (`/models/decision_tree_graph/`).
  - MedPrompt (`/models/medprompt/`).
- `tests/`: Includes test data and unit tests for various components.

For running evaluation, one needs to pass all the required set of arguments in the `eval_runner.py` script, which additionally gets a `config_path` to a specific YAML file with all the optional arguments (see `em/models/decision_tree_graph/config/runner.yaml` for instance).
For assistance on how to write your evaluation runners, please visit the readme file under `cdrc/common` folder or see the example given in `eval_runner.py`.

## Usage

To run the decision tree model:

```
python -m cdrc.profees_coding.em.eval_runner  --data-path <path_to_data_file> \
                                          --batch-size <batch_size> \
                                          --model-name <model_name> \
                                          --model-kwargs <model_kwargs> \
                                          --prompt-path <path_to_prompt_yaml_file> \
                                          --prompt-version <prompt_version> \
                                          --prompt-contributor <prompt_contributor_guid> \
                                          --run-name <run_name> \
                                          --output-dir <output_directory> \
```

## Testing

Run the unit tests using:

```
python -m unittest discover cdrc/profees_coding/em/tests
```
