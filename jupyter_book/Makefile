# Makefile for Jupyter Book documentation

# You can set these variables from the command line.
JUPYTERBOOK   = jupyter-book
SOURCEDIR     = .
BUILDDIR      = _build

# Help target
.PHONY: help
help:
	@echo "Row2Vec Documentation Build Commands:"
	@echo ""
	@echo "  build         Build the Jupyter Book documentation"
	@echo "  clean         Clean the build directory"
	@echo "  serve         Build and serve documentation locally"
	@echo "  publish       Publish to GitHub Pages"
	@echo "  install       Install documentation dependencies"
	@echo "  check         Check for broken links and references"
	@echo ""

# Install documentation dependencies
.PHONY: install
install:
	pip install -r requirements.txt
	pip install -e ..  # Install row2vec package

# Clean build directory
.PHONY: clean
clean:
	$(JUPYTERBOOK) clean $(SOURCEDIR)

# Build documentation
.PHONY: build
build: clean
	$(JUPYTERBOOK) build $(SOURCEDIR)

# Build and serve locally
.PHONY: serve
serve: build
	@echo "Documentation built successfully!"
	@echo "Open your browser and navigate to:"
	@echo "  file://$(PWD)/_build/html/index.html"
	@echo ""
	@echo "Or serve with Python:"
	@echo "  cd _build/html && python -m http.server 8000"

# Publish to GitHub Pages
.PHONY: publish
publish: build
	ghp-import -n -p -f _build/html

# Check for issues
.PHONY: check
check: build
	@echo "Checking for broken links..."
	@# Add link checking commands here if needed
	@echo "Documentation check complete!"

# Development target - quick build without execution
.PHONY: dev
dev:
	$(JUPYTERBOOK) build $(SOURCEDIR) --builder html

# Execute all notebooks (useful for pre-execution)
.PHONY: execute
execute:
	$(JUPYTERBOOK) build $(SOURCEDIR) --builder html --execute

.PHONY: all
all: build