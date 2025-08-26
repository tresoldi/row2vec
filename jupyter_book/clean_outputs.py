#!/usr/bin/env python3
"""
Update code cells to suppress warnings and logging output more effectively.
"""

import os
import re

# Files to update
EXECUTABLE_FILES = [
    "intro.md",
    "quickstart.md",
    "titanic_example.md",
    "adult_example.md",
    "housing_example.md",
    "advanced_features.md",
    "installation.md",
    "test_simple.md",
]

def update_code_cells(filename):
    """Update code cell tags to suppress all unwanted output."""
    if not os.path.exists(filename):
        print(f"⚠ File not found: {filename}")
        return

    with open(filename) as f:
        content = f.read()

    # Pattern to find code cells with basic tags
    pattern = r"```{code-cell} python\n:tags: \[remove-stderr\]"
    replacement = r"```{code-cell} python\n:tags: [remove-stderr, remove-warnings]"

    # Replace the pattern
    new_content = re.sub(pattern, replacement, content)

    # Also update cells without any tags to include clean output tags
    pattern_no_tags = r"```{code-cell} python\n(?!:tags:)"
    replacement_no_tags = r"```{code-cell} python\n:tags: [remove-stderr, remove-warnings]\n"

    new_content = re.sub(pattern_no_tags, replacement_no_tags, new_content)

    if new_content != content:
        with open(filename, "w") as f:
            f.write(new_content)
        print(f"✓ Updated code cell tags in {filename}")
    else:
        print(f"- No changes needed in {filename}")

def main():
    """Update all executable markdown files."""
    print("Updating code cell tags for cleaner output...")

    for filename in EXECUTABLE_FILES:
        update_code_cells(filename)

    print("\nDone! Code cells now have improved output filtering.")

if __name__ == "__main__":
    main()
