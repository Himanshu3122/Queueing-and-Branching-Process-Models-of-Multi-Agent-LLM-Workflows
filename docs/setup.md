# Setup

## Requirements

- **Python:** 3.10 or newer. Tested locally on 3.13.17; CI targets 3.11–3.13.
- **Hardware:** CPU only; no GPU.
- **Runtime dependencies:** `numpy`, `scipy`, `pydantic>=2`, `pyyaml`.
- **Optional API:** `starlette`, `uvicorn`.
- **Development:** `pytest`, `ruff`.

## From a clean environment

```bash
git clone <your-fork-url> agentq && cd agentq
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[api,dev]"          # or: pip install -r requirements-dev.txt
```

Pinned versions (the ones the test suite ran with) are in `requirements*.txt`. `pyproject.toml` uses minimum versions.

## Verify the installation

```bash
agentq --help
make test          # 53 tests, ~10 s
make smoke         # end-to-end smoke workflow, ~5 s
```

## Running without installing

All commands also work from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m agentq predict --profile P3 --rho 0.8
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

The CLI's default config paths (`configs/profiles.yaml`, `configs/experiments.yaml`) are relative to the repository root. From elsewhere, pass `--profiles` / `--config`.

## Offline installs

With an internal package mirror, set `PIP_INDEX_URL`. The core needs only NumPy, SciPy, Pydantic and PyYAML. If PyPI is unreachable but these are already installed, `pip install -e . --no-deps --no-build-isolation` installs the package itself.
