# Security Policy

## Supported versions

Row2Vec is under active development. Security fixes are applied to the latest
released version on PyPI. Please make sure you are running the most recent
release before reporting an issue.

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

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

- **Model deserialization.** A saved model is a single `.r2v` file (a zip
  archive). **Loading it does not execute code from the file**: there is no loader
  script and no pickle. scikit-learn objects are read with
  [skops](https://skops.readthedocs.io), restricted to an exact allow-list of
  types (`row2vec.serialization.TRUSTED_TYPES`); the encoder of a neural model is
  read in Keras's own format with `safe_mode=True` and only `InputLayer`, `Dense`
  and `Dropout` layers permitted; everything else is JSON. A file that names
  anything off the list raises `ModelFormatError` and is not loaded. The archive
  is also checked for unexpected members and per-member checksums.
  A custom `text_encoder` is a Python callable, so it is never written to the
  file: `load_model(path, text_encoder=fn)` takes it from the caller, which means
  the code that runs is the code you passed, not anything in the archive.

  What this does **not** give you:
  - *Authenticity.* The checksums detect corruption, not tampering by someone who
    rewrites the file and its manifest. Load models only from sources you trust,
    as you would any data file you are about to base decisions on.
  - *Protection from a vulnerable dependency.* The guarantee rests on skops, h5py
    and Keras behaving as documented; keep them updated.
  - *Privacy.* A model contains the category values and summary statistics its
    preprocessing learned from your data (not the rows themselves). Treat it as
    derived from the training data when you share it.

  The previous script-and-pickle format executed the script when loaded. It was
  never released, and `load_model` refuses `.py` and `.pkl` paths without
  running them.
- **Untrusted DataFrames and configuration files.** Adversarial column names,
  dtypes, or YAML configuration passed to the CLI.

Reports in those areas are especially welcome.
