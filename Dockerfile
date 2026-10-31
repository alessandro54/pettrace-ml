# The pinned evaluation environment (Linux, Python 3.11, versions from uv.lock); the published
# results reproduce in it (README.md). The code is mounted at /repo at run time, so a run always uses the checked-out commit.
FROM ghcr.io/astral-sh/uv:0.11.22 AS uv
FROM python:3.11.15-slim-bookworm
COPY --from=uv /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /repo
COPY pyproject.toml uv.lock .python-version ./
# GROUPS="--group figures --group notebooks --group export" builds the registration image (torch + transformers).
ARG GROUPS="--group figures --group notebooks"
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project $GROUPS
ENV PATH=/venv/bin:$PATH PYTHONPATH=/repo/src GIT_PYTHON_REFRESH=quiet
ENTRYPOINT ["python"]
