#!/bin/bash

# Row2Vec Documentation Build Script
# This script ensures consistent documentation generation following Jupyter Book best practices

echo "Building Row2Vec documentation..."
echo "=================================="

# Clean any existing build artifacts
echo "Cleaning previous builds..."
rm -rf _build/

# Build the documentation using standard Jupyter Book command
# This will create _build/html/ with all documentation files
echo "Generating documentation..."
jupyter-book build .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Documentation build completed successfully!"
    echo "📂 HTML files are located in: _build/html/"
    echo "🌐 Open _build/html/index.html in your browser to view the documentation"
    echo ""
    echo "File path for direct access:"
    echo "file://$(pwd)/_build/html/index.html"
else
    echo ""
    echo "❌ Documentation build failed!"
    echo "Please check the error messages above."
    exit 1
fi