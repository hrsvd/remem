# Contributing

Thanks for your interest in Remem!

Remem follows semantic versioning and keeps its stable public APIs compatible.

## Philosophy

- Simplicity over cleverness
- Correctness before optimization
- Measure before optimizing
- Keep public APIs stable

## Development

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Install the development extra with `pip install -e ".[dev]"`
5. Run `python -m pytest`, `python -m ruff check .`,
   `python -m ruff format --check .`, and `python -m mypy remem`
6. For packaging changes, run `python -m build` and
   `python -m twine check dist/*`
7. Open a pull request
