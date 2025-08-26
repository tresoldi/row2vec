#!/usr/bin/env python3
"""
Fix MyST markdown files to be executable by adding proper notebook metadata.
"""

import os

# Notebook metadata to add
NOTEBOOK_METADATA = """---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

"""

# Files that need executable code
EXECUTABLE_FILES = [
    "titanic_example.md",
    "adult_example.md",
    "housing_example.md",
    "advanced_features.md",
    "installation.md",  # Has one verification code block
]

def fix_file(filename):
    """Add notebook metadata to a markdown file if it doesn't have it."""
    with open(filename) as f:
        content = f.read()

    # Check if it already has metadata
    if content.startswith("---\njupytext:"):
        print(f"✓ {filename} already has metadata")
        return

    # Add metadata at the beginning
    new_content = NOTEBOOK_METADATA + content

    with open(filename, "w") as f:
        f.write(new_content)

    print(f"✓ Added metadata to {filename}")

def main():
    """Fix all executable markdown files."""
    print("Adding Jupyter notebook metadata to MyST files...")

    for filename in EXECUTABLE_FILES:
        if os.path.exists(filename):
            fix_file(filename)
        else:
            print(f"⚠ File not found: {filename}")

    print("\nDone! Files are now ready for execution.")

if __name__ == "__main__":
    main()
