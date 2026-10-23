# Common tasks. Run from the repository root.
PY ?= python3
export PYTHONPATH := src

.PHONY: install install-dev lint format test test-unit test-integration test-smoke smoke reproduce compare serve docker clean

install:            ## install the package with the API extra
	$(PY) -m pip install -e ".[api]"

install-dev:        ## install with test/lint tools
	$(PY) -m pip install -e ".[api,dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

test:               ## all tests (unittest runner; pytest also works: `pytest`)
	$(PY) -m unittest discover -s tests -t .

test-unit:
	$(PY) -m unittest discover -s tests/unit -t .

test-integration:
	$(PY) -m unittest discover -s tests/integration -t .

test-smoke:
	$(PY) -m unittest discover -s tests/smoke -t .

smoke:              ## end-to-end smoke workflow (few seconds)
	$(PY) scripts/smoke_test.py

reproduce:          ## ALL paper experiments in full mode (tens of minutes on 2 CPU cores)
	$(PY) -m agentq run all --mode full

compare:            ## compare full-mode results with the paper's raw outputs
	for e in load_sweep fork_join bottleneck ablation trees cv_sensitivity burst_bracket capacity depth_sweep branching burst_failure; do \
		$(PY) -m agentq compare $$e --summary; done

serve:
	$(PY) -m agentq serve --host 127.0.0.1 --port 8000

docker:
	docker build -t agentq:latest .

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info results/smoke
