# Medical Current Procedural Terminology (CPT) Coding Project

This project is designed to automate the process of assigning CPT codes to medical encounters, particularly focusing on Evaluation and Management (E/M) services. It includes various components for data processing, model implementation, and evaluation.


## Key Components
Please follow the below conventions for the project structure. Each subproject (e.g. CPT EM Coding `em`) comprises of the following directories:
- `mcp/`: Contains mcp core code.
- `mcp/tools/`: Contains mco tools


For running evaluation, one needs to pass all the required set of arguments in the `eval_runner.py` script, which additionally gets a `config_path` to a specific YAML file with all the optional arguments (see `em/models/decision_tree_graph/config/runner.yaml` for instance).
For assistance on how to write your evaluation runners, please visit the readme file under `cdrc/common` folder or see the example given in `eval_runner.py`.

## Usage

## AI Mode MCP Tools and Skills

AI mode development uses three moving parts:

- `src/profees_coding/mcp/`: the FastMCP server that exposes clinical data,
  document, ICD-10, and inpatient coding tools to an AI agent.
- `profees_coding_skills/`: the packaged PCA skills and UI schemas that teach
  the agent how to call the MCP tools and render coding results.
- A consumer AI-mode/PCA persona, such as a coder persona in a sibling
  AI-mode repository, that mounts the skills package and connects to the local
  MCP endpoint.

The MCP server is documented in
[`src/profees_coding/mcp/README.md`](src/profees_coding/mcp/README.md). For
local AI-mode work, run the MCP server, then start the PCA persona and UI shell
from their respective repositories. The local MCP endpoint is:

```text
http://127.0.0.1:9000/mcp
```

Use the health endpoint to verify the server before connecting the agent:

```bash
curl http://127.0.0.1:9000/health
```


Then run `uv sync` in the consumer persona and restart the PCA runtime.

When changing skills, validate the package from the repo root:

```bash
cd profees_coding_skills
uv sync --extra dev
uv run pytest
uv run python scripts/check_version.py 0.1.8
uv build
```

Do not commit `.env` files, AI Factory credentials, MAX session IDs, MRNs, FINs,
or other environment-specific clinical identifiers.

## AgentLens Observability (Local Eval Only)

To run the decision tree model:

```
TODO
```

Prompt files are resolved through the ProFees prompt library. Keep `prompt_path`
pointing at the legacy model config prompt path, such as
`.../config/prompts.yaml`; the resolver uses that path to find scoped prompt
bundles under `config/{care_setting}/{client_name}/{specialty}/prompts.yaml`.
Supported care settings are `ambulatory`, `ed`, and `inpatient`; the current
default client is `general`; supported specialties are `primary_care` and
`oncology`. The default route is
`config/ambulatory/general/primary_care/prompts.yaml`, followed by the legacy
path.

Each scoped prompt file should include top-level routing metadata and
per-version metadata. Use explicit placeholders when release or base-model
details are not yet documented:

```yaml
metadata:
  model_design: medprompt_detailed_multi
  care_setting: ambulatory
  client_name: general
  specialty: primary_care

encounter_type:
  v1:
    status: legacy
    release: unknown
    basemodel: unspecified
    system: |
      System prompt text...
    user: |
      User prompt text...

scalar_template_bank:
  v1:
    status: legacy
    release: unknown
    basemodel: unspecified
    prompt: |
      Scalar template text...
```

`status`, `release`, and `basemodel` are stripped before prompt rendering, so
graph nodes receive the same prompt shapes they used before migration.

## Testing

Run the unit tests using:

```
python -m unittest discover cdrc/profees_coding/em/tests
```
