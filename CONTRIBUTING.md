# Contributing

Thanks for helping improve agentq. This is a research codebase; correctness and traceability to the paper matter more than features.

## Workflow

1. Fork and create a branch.
2. `pip install -e ".[api,dev]"`
3. Make changes with tests. Run `make lint test smoke` before opening a PR.
4. In the PR description, say which paper section/equation the change relates to, and whether it changes any reproduced number.

## Ground rules

- **Do not change seeded behavior silently.** The simulators reproduce the paper's outputs bit-for-bit (pinned by `tests/integration/test_reference_regression.py`). If a change alters random-number draw order or results, update the golden values in the same PR and explain why.
- **Keep `full` configs faithful.** `configs/experiments.yaml` `full` blocks encode the paper's protocol. Add new settings as new experiments or modes.
- **Label data honestly.** Synthetic data must be marked synthetic; reproduced results must carry provenance (the runner adds it).
- **Record assumptions.** New approximations or inferred behavior go in `docs/assumptions.md`, and mapping changes in `docs/paper_implementation_mapping.md`.
- **Style.** Ruff for linting and formatting (line length 120), type hints, and docstrings on public functions.

## Reporting issues

Include the command, the full output, `python --version`, and `pip freeze | grep -E "numpy|scipy|pydantic"`.
