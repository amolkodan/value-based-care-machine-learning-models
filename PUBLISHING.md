# Publishing `carevalue-claims-ml` to PyPI

This repository ships **one** distribution on PyPI named **`carevalue-claims-ml`**. It includes the umbrella **VBC Intelligence OS** import namespaces (`vbc_intel_*`) and the existing `carevalue_claims_ml` package. Separate PyPI projects per sublibrary are optional and not required for users to `pip install` once.

## Prerequisites

1. A [PyPI](https://pypi.org/account/register/) account (and optionally [TestPyPI](https://test.pypi.org/account/register/) for a dry run).
2. The package name **`carevalue-claims-ml`** must be **available** on PyPI. If it is taken, change `name` in `pyproject.toml` and try again.
3. Your GitHub repository pushed to `origin` (this doc assumes `amolkodan/value-based-care-machine-learning-models`).

## Option A — GitHub Actions + PyPI Trusted Publishing (recommended)

No long-lived API token stored in GitHub secrets if you use OIDC.

### 1) Configure Trusted Publisher on PyPI

1. Log in to [pypi.org](https://pypi.org).
2. Go to **Account settings** → **Publishing** → **Add a new pending publisher**.
3. Choose **GitHub** as the publisher.
4. Set:
   - **PyPI project name**: `carevalue-claims-ml` (must match `[project].name` in `pyproject.toml`).
   - **Owner**: `amolkodan`
   - **Repository name**: `value-based-care-machine-learning-models`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: `pypi` (must match the workflow’s `environment: name: pypi`).

Save the pending publisher on PyPI.

### 2) Create the `pypi` environment in GitHub

1. In your GitHub repo: **Settings** → **Environments** → **New environment** → name it **`pypi`**.
2. Optional: add **Required reviewers** so uploads only run after approval.

### 3) Release a version

1. Ensure `version` in `pyproject.toml` matches the tag you want (e.g. `0.1.0`).
2. Commit and push to `main`.
3. In GitHub: **Releases** → **Create a new release** → create a tag like `v0.1.0` → **Publish release**.

The workflow [`.github/workflows/publish-pypi.yml`](.github/workflows/publish-pypi.yml) will build the sdist and wheel and upload them to PyPI when the release is **published**.

### 4) Verify

- Project page: `https://pypi.org/project/carevalue-claims-ml/`
- Install: `pip install carevalue-claims-ml`

## Option B — Manual upload from your laptop

Use this for first-time testing or if you prefer API tokens.

### 1) Create an API token

On PyPI: **Account settings** → **API tokens** → create a token scoped to this project (or whole account for first upload).

### 2) Build and check locally

```bash
cd /path/to/value-based-care-machine-learning-models
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip build twine
python -m build
twine check dist/*
```

### 3) Upload

**TestPyPI (dry run):**

```bash
twine upload --repository testpypi dist/*
```

**Production PyPI:**

```bash
twine upload dist/*
```

When prompted, username `__token__` and password your PyPI API token (including `pypi-` prefix).

## After publishing

- Tell users: `pip install carevalue-claims-ml`
- CLIs: `carevalue-ml` and `vbc-intel`
- Imports: `carevalue_claims_ml`, `vbc_intel_core`, `vbc_intel_episodes`, etc.

## Version bumps

1. Update `version` in [`pyproject.toml`](pyproject.toml) and [`src/carevalue_claims_ml/__init__.py`](src/carevalue_claims_ml/__init__.py) (if you keep them in sync).
2. Add an entry to [`CHANGELOG.md`](CHANGELOG.md).
3. Tag `vX.Y.Z` and create a GitHub Release (for Option A).

## Separate PyPI packages per sublibrary

Publishing five independent projects (e.g. `vbc-intel-core`, `vbc-intel-episodes`) requires either multiple `pyproject.toml` trees or a monorepo build tool. That is **not** set up in this repo yet; the current design is one wheel containing all namespaces.
