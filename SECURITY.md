# Security Policy

agentq is a research engineering repository. It has **not** been security-audited and makes no production-security claims.

## Scope and posture

- The HTTP API has **no authentication** and is meant for local or internal use. Put it behind an authenticating reverse proxy if exposed.
- Inputs are validated with Pydantic. Grid sizes per request are bounded (`AGENTQ_MAX_GRID_POINTS`) to limit CPU and memory use.
- No secrets are needed or stored. `.env` is git-ignored; `.env.example` holds only non-secret settings.
- The code does not execute shell commands or load pickled data. Configs are read with `yaml.safe_load`.

## Model-risk note

Capacity numbers are model predictions that the paper shows to be optimistic near saturation. Do not use them as the sole basis for production provisioning; heed validity flags and validate against measurements.

## Reporting a vulnerability

Please report privately via the repository's security advisory feature rather than a public issue. Include steps to reproduce and the affected version.
