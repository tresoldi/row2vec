# Building Row2Vec Documentation

This document explains how to build the Row2Vec documentation consistently.

## Quick Start

The simplest way to build the documentation:

```bash
# Build documentation (creates _build/html/)
jupyter-book build .
```

## Build Scripts

### Using the Build Script (Recommended)

```bash
# Clean build with status messages
./build_docs.sh
```

### Using Makefile

```bash
# Build with clean
make build

# Quick development build
make dev

# Build and show local serving options
make serve
```

## Directory Structure

After building, the documentation will be available in:

```
_build/
├── html/                 # Generated HTML documentation
│   ├── index.html       # Main documentation page
│   ├── adult_example.html
│   ├── housing_example.html
│   ├── titanic_example.html
│   └── ...
└── jupyter_execute/     # Executed notebooks and images
    ├── *.ipynb          # Converted MyST notebooks
    └── *.png            # Generated plots/images
```

## Configuration

The build behavior is controlled by `_config.yml`:

- **Output Directory**: Always builds to `_build/html/`
- **Execution**: Notebooks are executed automatically if no outputs exist
- **Error Handling**: Allows errors for demonstration purposes
- **Output Cleaning**: Removes warnings and stderr for clean documentation

## Best Practices

1. **Always use the standard command**: `jupyter-book build .`
2. **Clean builds**: Use `make clean` or `./build_docs.sh` for clean rebuilds
3. **Check outputs**: Verify generated HTML in `_build/html/index.html`
4. **Consistent paths**: Never specify custom output directories

## Troubleshooting

### Multiple Build Directories

If you see inconsistent directories like `_build/_build/`, clean everything:

```bash
rm -rf _build/
jupyter-book build .
```

### Build Errors

1. Check Python environment has required packages
2. Ensure `suppress_minimal.py` exists
3. Verify data files are present in `../data/`
4. Use `jupyter-book build . --verbose` for detailed error messages

### Cache Issues

Clean the execution cache if needed:

```bash
jupyter-book clean . --cache
jupyter-book build .
```

## Output Location

✅ **Correct**: Documentation builds to `_build/html/`
❌ **Incorrect**: `_build/_build/html/`, `docs/`, or other custom paths

The `_build/html/` structure is Jupyter Book's standard and ensures consistency across all builds.
