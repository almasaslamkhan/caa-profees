"""Physician persona skill package.

This package is mounted via `pca.skills` in the persona config as a
packaged resource root. External skills are copied into the ``bundled``
directory during the build step (see ``skill_bundler.bundle_skills``) so
they ship with the persona artifact without living in the source tree.
"""
