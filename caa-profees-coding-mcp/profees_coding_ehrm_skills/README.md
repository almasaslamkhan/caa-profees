# ProFees Coding EHRM Skills

Resource-only Python package for the EHRM ProFees ambulatory-coding UI-event
skill. The wheel contains the `ambulatory-coding` skill and its
`custom.codingAgentRecodeCustomEHRMEvent` event trigger.

## Build

From this directory, build a versioned wheel with:

```bash
uv sync --extra dev
uv run python scripts/check_version.py 0.0.10763
uv build
```

The build creates these publishable artifacts in `dist/`:

```text
profees_coding_ehrm_skills-<version>-py3-none-any.whl
profees_coding_ehrm_skills-<version>.tar.gz
```

## Install In An EHRM Persona

After publishing, add the package as a persona dependency and mount its skill
resources:

```toml
[project]
dependencies = [
  "profees-coding-ehrm-skills>=0.0.10763",
]

[tool.proactive_pca.skills]
mounts = [
  { package = "profees_ambulatory_coding_ehrm_skills", resource_path = "skills" },
]
```

For local development, use an editable source instead:

```toml
[project]
dependencies = [
  "profees-coding-ehrm-skills",
]

[tool.uv.sources]
profees-coding-ehrm-skills = { path = "../caa-profees-coding-mcp/profees_coding_ehrm_skills", editable = true }
```

Run `uv sync` in the consuming persona after changing its dependencies or skill
mounts, then restart the PCA runtime.
