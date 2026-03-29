## Contributing

Thanks for contributing to `carevalue-claims-ml`.

### Development setup

- Create and activate a virtual environment.
- Install dependencies with `pip install -e ".[dev]"`.
- Run checks with `ruff check src tests` and `pytest -q`.

### Pull request expectations

- Keep changes scoped and documented.
- Add or update tests for behavior changes.
- Update README or model docs when adding new model capabilities.
- Avoid committing credentials, secrets, or real PHI.

### Commit style

- Prefer clear, outcome-oriented commit messages.
- Keep commits focused by feature area when possible.

### Data safety

- Use synthetic or de-identified data only.
- Never commit raw production patient data.
