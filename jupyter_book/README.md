# Row2Vec Documentation

This directory contains the source files for Row2Vec's documentation, built using [Jupyter Book](https://jupyterbook.org/).

## 🌐 Live Documentation

The documentation is deployed at: **https://evotext.github.io/row2vec/**

## 🏗️ Building Documentation

### Prerequisites

```bash
pip install -r requirements.txt
pip install -e ..  # Install row2vec package
```

### Local Build

```bash
# Quick build for development
./build_docs.sh

# Manual build
jupyter-book build .

# Clean build (removes cache)
jupyter-book build . --all
```

## Documentation Structure

- `intro.md` - Main introduction and overview
- `installation.md` - Installation instructions
- `quickstart.md` - 5-minute getting started guide
- `*_example.md` - Detailed examples with executable code
- `advanced_features.md` - Expert features and configurations
- `cli_guide.md` - Command-line interface documentation
- `api_reference.md` - Complete API documentation
- `changelog.md` - Version history and changes

## Features

- **Executable Code**: All examples run real code with outputs
- **MyST Markdown**: Rich markdown with Jupyter-style code cells
- **Auto-generated API**: Sphinx integration for API documentation
- **GitHub Pages**: Easy deployment with GitHub Actions

## Development

### Building Locally

```bash
# Full build with execution
make build

# Quick build without execution (for development)
make dev

# Clean and rebuild
make clean && make build
```

### Adding New Content

1. Create new `.md` files with MyST syntax
2. Add to `_toc.yml` table of contents
3. Use `{code-cell}` blocks for executable code
4. Build and test locally

### Code Cell Format

```markdown
{code-cell} python
:tags: [remove-stderr]

# Your executable Python code here
import pandas as pd
result = some_computation()
print(result)
```

## Publishing

### GitHub Pages (Automatic)

Documentation is automatically built and published via GitHub Actions on push to main branch.

### Manual Publishing

```bash
make publish
```

## Configuration

Key configuration in `_config.yml`:
- Execution settings (currently disabled for pre-executed notebooks)
- MyST extensions for advanced markdown features
- Sphinx configuration for API documentation
- Repository and launch button settings

## Troubleshooting

**Import errors when building:**
```bash
# Ensure row2vec package is installed
pip install -e ..
```

**Execution errors:**
- Check that all required data files exist in `../data/`
- Verify all imports are available
- Use `:tags: [remove-stderr]` to hide non-critical warnings

**GitHub Pages not updating:**
- Check GitHub Actions logs
- Ensure `gh-pages` branch exists
- Verify repository settings for Pages source

## Architecture Notes

This documentation system:
- Uses Jupyter Book for static site generation
- MyST markdown for executable content (instead of .ipynb files)
- Pre-executed outputs committed to repository
- Auto-generated API docs via Sphinx
- Flat directory structure for simplicity
- GitHub Pages for hosting

The design prioritizes:
1. **Simplicity**: Minimal tooling complexity
2. **Maintainability**: Easy to update and refactor
3. **Executable accuracy**: Real outputs from real code
4. **Fast builds**: Pre-executed content
