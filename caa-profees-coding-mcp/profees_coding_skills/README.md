# Profees Coding Skills

Reusable Proactive PCA skill package for professional-fees coding workflows.

This repository is intentionally resource-only. It packages `SKILL.md` files and
supporting assets so PCA personas can mount them after installing the wheel from
OCI Artifactory.

## Published Skills

- `ambulatory-coding`: A2UI schema resources for launching the ambulatory
  generated-codes canvas plugin.

The legacy `src/skills_package/skills/profees-coding` source tree remains in
the repository for existing source-level development and tests, but it is
intentionally excluded from the published wheel. The generic `skills_package`
namespace is reserved for PCA-local skills and must not be distributed by this
package.

## Install In A PCA Persona

After this package is published to OCI Artifactory, add the distribution as a
persona dependency:

```toml
[project]
dependencies = [
  "profees-skills>=0.0.10763",
]
```

Then mount the package resources:

```toml
[tool.proactive_pca.skills]
mounts = [
  { package = "profees_ambulatory_coding_skills", resource_path = "skills" },
]
```

Alternatively, enable it through persona config:

```yaml
pca:
  external_skills:
    enabled: true
    mounts:
      - { package: "profees_ambulatory_coding_skills", resource_path: "skills" }
```

Run `uv sync` in the consumer persona after adding the dependency, then restart
the PCA runtime.

## Use Local Checkout During Development

For local UI/schema iteration, point the consumer persona to this checkout
instead of a published package version:

```toml
[project]
dependencies = [
  "profees-skills",
]

[tool.uv.sources]
profees-skills = { path = "../clinical-rev-cycle/profees_coding_skills", editable = true }
```

Keep the skill mount the same:

```toml
[tool.proactive_pca.skills]
mounts = [
  { package = "profees_ambulatory_coding_skills", resource_path = "skills" },
]
```

Then run `uv sync` in the consumer persona and restart the PCA runtime. The
`package://profees_ambulatory_coding_skills/...` schema paths will resolve to
the local editable package.

## Develop

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/check_version.py 0.0.10763
uv build
```

Verify the wheel contains only the published ambulatory skill resources:

```bash
uv run python scripts/check_wheel_contents.py dist
```

## Publish

The preferred publish path is OCI Build, not a local developer upload. The root
`ocibuild.conf` has a dedicated `package_profees_skills` step that
tests and builds this package, followed by `publish_profees_skills_dev`
which publishes to `ohai-cda-shared-library-dev-pypi-local` through the build
service account.

Keep `project.version` in `profees_coding_skills/pyproject.toml` as a literal
PEP 440 version. OCI Build publishes immutable build-numbered artifacts by
running `scripts/check_version.py 0.0.${BLD_NUMBER}`; that command stamps the
package version inside the disposable build workspace before `uv build` when
the expected version has the `0.0.<build-number>` form.

If a repo owner explicitly grants local deploy access, the equivalent manual
publish command is:

```bash
uv publish --trusted-publishing never \
  --publish-url https://artifactory-builds.oci.oraclecorp.com/api/pypi/ohai-cda-shared-library-dev-pypi-local \
  dist/*
```

Use immutable versions. Bump `project.version` before each publish.
