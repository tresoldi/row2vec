# Security Policy

## Supported versions

Row2Vec is under active development. Security fixes are applied to the latest
released version on PyPI. Please make sure you are running the most recent
release before reporting an issue.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a vulnerability

If you believe you have found a security vulnerability, please report it
privately rather than opening a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/tresoldi/row2vec/security/advisories/new)
  (Security → Report a vulnerability), or
- Email the maintainer at row2vec@tresoldi.org.

Please include a description of the issue, the affected version(s), and, if
possible, a minimal reproduction. You can expect an initial acknowledgement
within a reasonable timeframe, and we will keep you informed as the report is
investigated and resolved.

Because Row2Vec is a data-science library with no network or authentication
surface, the most likely concerns are around untrusted input. Two areas are
worth calling out:

- **Model deserialization.** `load_model()` restores a saved pipeline, which
  includes unpickling preprocessing objects. Only load models from sources you
  trust; a malicious model file can execute arbitrary code, as with any
  pickle-based format.
- **Untrusted DataFrames and configuration files.** Adversarial column names,
  dtypes, or YAML configuration passed to the CLI.

Reports in those areas are especially welcome.
