# Repository Guidelines

## Project Structure & Module Organization
- `src/hescape/` contains the Python package. Key subpackages: `models/image_models/` for slide encoders, `models/dnameth_models/` for CpGPT wrappers, `data_modules/` for Lightning data modules, and `modules/` for training orchestration.
- `tests/` stores pytest suites; mirror module paths (e.g., `tests/data_modules/test_dnameth_wsi.py`) when adding coverage.
- `experiments/` holds Hydra/Lightning configs and helper scripts; `docs/` contains the Sphinx site; large slide assets live under `data/` and should remain git-ignored or symlinked.

## Build, Test, and Development Commands
- Activate the maintained environment with `conda activate gigapath`, then `hatch env create` on first setup.
- Use `hatch shell` for a managed dev session; `hatch run docs:build` renders documentation; `hatch build` prepares a wheel.
- Run `hatch test` (or `hatch test --all`) for the full pytest matrix; use `pytest -k <pattern>` for targeted checks; `pre-commit run --all-files` enforces style before pushing.

## Coding Style & Naming Conventions
- Format with `ruff format` (120 char lines) and lint with `ruff check`; both run via pre-commit hooks.
- Prefer snake_case for modules/functions, PascalCase for classes, and descriptive config filenames in `experiments/`.
- Write NumPy-style docstrings and include type hints; keep notebooks in `docs/notebooks` or `notebooks/` with executed outputs trimmed before commit.

## Testing Guidelines
- Place new tests alongside code counterparts under `tests/`; name files `test_<module>.py` and functions `test_<behavior>`.
- Respect `pytest` markers already present (e.g., remove skips once the behavior stabilizes) and avoid silent xfails—`xfail_strict` is enabled.
- Use `coverage run -m pytest` when touching core trainers or data pipelines and open a coverage report for regressions.

## Commit & Pull Request Guidelines
- Base branch history uses short imperative messages (e.g., "change cpgpt"); prefer `component: action` (e.g., `models: add dna-meth head`).
- Each PR should describe motivation, dataset/experiment touch points, test evidence, and links to tracking issues; attach result tables or paths from `hescape_results.md` when applicable.
- Sync with pre-commit.ci before requesting review and rerun critical experiments referenced in the PR description.

## Data & Experiment Notes
- Raw slides and methylation tables are large; store them outside git and point configs to local paths via Hydra overrides.
- Keep generated figures under `figures/` and pipeline logs under `experiments/` or `output.log`; clean ephemeral artifacts with `git clean -fdX docs` as needed.
