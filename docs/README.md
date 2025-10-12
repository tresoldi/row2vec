# Row2Vec Documentation

This directory contains executable tutorial documentation generated with [Nhandu](https://github.com/tresoldi/nhandu).

## Structure

```
docs/
├── tutorial_*.py        # Executable Python tutorial sources
├── tutorial_*.html      # Generated HTML documentation (gitignored)
├── figures/             # Tutorial figures and plots
├── API_REFERENCE.md     # Complete API documentation
└── README.md            # This file
```

## Building Documentation

### Prerequisites

Install nhandu and dependencies:

```bash
pip install -e .[dev]  # Installs nhandu and other dev tools
```

### Generate HTML Documentation

Use the Makefile target:

```bash
make docs
```

Or manually:

```bash
nhandu docs/tutorial_1_quickstart.py --format html -o docs/tutorial_1_quickstart.html
nhandu docs/tutorial_2_advanced.py --format html -o docs/tutorial_2_advanced.html
```

### Clean Generated Documentation

```bash
make docs-clean
```

## Tutorial Files

Tutorial files use Nhandu's literate programming format:

- `#'` prefix: Markdown content (documentation)
- Regular code: Executed Python code with output capture
- `#| hide` / `#|`: Hidden code blocks (imports, setup, etc.)

Example:

```python
#' # Tutorial Title
#'
#' This is documentation in markdown.

import pandas as pd

# This code executes and shows output
data = pd.DataFrame({'A': [1, 2, 3]})
print(data)

#' ## Section 2
#'
#' More documentation...

#| hide
# This code runs but doesn't appear in output
import warnings
warnings.filterwarnings('ignore')
#|
```

## Available Tutorials

1. **tutorial_1_quickstart.py** - Get started with Row2Vec in 5 minutes
2. **tutorial_2_advanced.py** - Advanced features (architecture search, imputation, serialization)

## Adding New Tutorials

1. Create `docs/tutorial_N_name.py` following the Nhandu format
2. Run `make docs` to generate HTML
3. Tutorials are automatically discovered by the Makefile

## Migration from Jupyter Book

This documentation was migrated from Jupyter Book (in `jupyter_book/`) to Nhandu for:

- **Executable documentation**: Code runs and is tested with every build
- **Simpler toolchain**: Just Python files, no Jupyter notebooks
- **Version control friendly**: Plain Python files in git
- **Integrated testing**: Tutorials serve as integration tests

The old `jupyter_book/` directory is retained for reference but is no longer the primary documentation source.

## Documentation Formats

### Nhandu Tutorials (Primary)
- **Location**: `docs/tutorial_*.py`
- **Format**: Executable Python with `#'` markdown comments
- **Output**: Self-contained HTML files
- **Advantages**: Executable, testable, version-control friendly

### API Reference
- **Location**: `docs/API_REFERENCE.md`
- **Format**: Markdown
- **Purpose**: Complete function and class reference

## Contributing

When adding new features:

1. Add usage examples to appropriate tutorials
2. Update `API_REFERENCE.md` with new functions/classes
3. Run `make docs` to verify examples execute correctly
4. Generated HTML files are gitignored - only commit `.py` sources

## External Documentation

For the full documentation site (including Jupyter Book content), see:
- https://evotext.github.io/row2vec/

## Questions?

- See the [main README](../README.md) for project overview
- Check [CHANGELOG.md](../CHANGELOG.md) for recent changes
- Report issues at https://github.com/evotext/row2vec/issues
