# The exact environment the published results were produced in (Linux, Python 3.11, versions from
# uv.lock). The code is mounted at /repo at run time, so a run always uses the checked-out commit.
FROM ghcr.io/astral-sh/uv:0.5.11-python3.11-bookworm-slim
ENV UV_PROJECT_ENVIRONMENT=/venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /repo
COPY pyproject.toml uv.lock ./
# GROUPS="--group figures --group notebooks --group export" builds the registration image (torch + transformers).
ARG GROUPS="--group figures --group notebooks"
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project $GROUPS
ENV PATH=/venv/bin:$PATH PYTHONPATH=/repo GIT_PYTHON_REFRESH=quiet
ENTRYPOINT ["python"]
