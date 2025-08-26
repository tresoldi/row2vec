#!/bin/bash

# Row2Vec Documentation Builder
# Single source of truth for all documentation generation
# Builds Jupyter Book documentation with executable MyST notebooks

set -e  # Exit on any error

echo "🚀 Building Row2Vec Documentation"
echo "=================================="
echo ""

# Navigate to jupyter_book directory
cd jupyter_book

# Set environment variables for cleaner TensorFlow output
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONHASHSEED=0

echo "📂 Working directory: $(pwd)"
echo "📝 Documentation format: Jupyter Book (MyST)"
echo ""

# Clean any existing build artifacts for fresh build
echo "🧹 Cleaning previous builds..."
rm -rf _build/

# Build the documentation using Jupyter Book
echo "📚 Generating documentation with executable notebooks..."
echo "   This may take a few minutes as code examples are executed..."
echo ""

# Build with error handling (no timeout to avoid interrupting long builds)
jupyter-book build . || {
    echo ""
    echo "❌ Documentation build failed!"
    echo ""
    echo "Common solutions:"
    echo "  1. Check for Python import errors in the notebook cells"
    echo "  2. Ensure all required packages are installed: pip install -e ."
    echo "  3. Verify data files exist in ../data/ directory"
    echo "  4. Check suppress_minimal.py exists and works"
    echo "  5. For kernel issues, try: jupyter-book clean . --all && jupyter-book build ."
    echo ""
    exit 1
}

# Check if build was successful
if [ -d "_build/html" ] && [ -f "_build/html/index.html" ]; then
    echo ""
    echo "✅ Documentation build completed successfully!"
    echo ""
    echo "📁 Documentation files:"
    echo "   Location: jupyter_book/_build/html/"
    echo "   Main page: _build/html/index.html"
    echo ""
    echo "🌐 To view documentation:"
    echo "   Local file: file://$(pwd)/_build/html/index.html"
    echo "   Or open: _build/html/index.html"
    echo ""
    echo "📊 Build statistics:"
    echo "   Total pages: $(find _build/html -name "*.html" | wc -l)"
    echo "   Size: $(du -sh _build/html | cut -f1)"
    echo ""
else
    echo ""
    echo "❌ Documentation build failed - output directory not created!"
    exit 1
fi

echo "🎉 Documentation ready for use!"