# Repo-root conftest: makes pytest prepend the repository root to sys.path,
# so `import signalmap` works under a bare `pytest` invocation (no editable
# install required). Keeps CI green regardless of install mode.
