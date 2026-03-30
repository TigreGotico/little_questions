# Maintainers Guide — little_questions

## Repository layout

| Path | Purpose |
|------|---------|
| `little_questions/` | Main package |
| `test/` | Unit tests (run without network/models) |
| `train_scripts/` | Model training and data cleaning |
| `docs/` | User and developer documentation |
| `examples/` | Usage examples |

## Development environment

```bash
uv pip install -e "."
uv run pytest test/ -v --cov=little_questions
```

## Releasing a new version

1. **Update version** in `little_questions/version.py` (source of truth).
   Also update `pyproject.toml` `version =` field to match.

2. **Retrain models** if `scikit-learn` version changed (pickle format is version-sensitive):
   ```bash
   uv run python train_scripts/train_en.py   # repeat for each language
   ```

3. **Update `MODEL2URL`** in `little_questions/models/__init__.py` to point to the new
   GitHub release tag. Update `LANG2MODEL` paths to match the new version suffix.

4. **Update `CHANGELOG.md`** — add a dated entry listing all changes.

5. **Tag the release** on GitHub (`v<version>`) and attach the `.pkl` model files as release assets.
   The `MODEL2URL` URLs must match the tag name exactly.

6. **Publish to PyPI** (when Phase 4 is complete):
   ```bash
   uv build
   uv publish
   ```

## CI

GitHub Actions workflow (planned — see ROADMAP.md Phase 3):
- Matrix: Python 3.10, 3.11, 3.12
- Steps: install deps → pytest → coverage report
- Tests run with mocked model zoo and xdg (no network)

## Model file naming convention

```
questions<N>_svm_<LANG>_<suffix>_<version>.pkl
```

| Field | Values |
|-------|--------|
| `N` | `52` (fine-grained) or `6` (coarse) |
| `LANG` | `EN`, `ES`, `PT`, `CA`, `FR`, `DE`, `IT` |
| `suffix` | `` (English) or `googtx` (translated training data) |
| `version` | Matches `little_questions/version.py` |

Example: `questions52_svm_EN_0.7.0.pkl`

## Dependency notes

- `JarbasModelZoo` — optional. Used to download Brill POS taggers for ES, PT, CA.
  If unavailable, those language pipelines fall back to n-gram/TF-IDF features only.
- `pyxdg` — required for model caching. Must be installed in the runtime environment.
- `nltk` — required for English sentence scorer (`pos_tag`, `word_tokenize`).
  NLTK data (`punkt`, `averaged_perceptron_tagger`) is downloaded on first EN model use.
- `requests` — used for model download. Not needed if models are pre-installed.
