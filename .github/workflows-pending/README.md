# Pending workflows — one `git mv` away from active

These three files are the CI for this branch. They are sitting here rather than
in `.github/workflows/` because the session that wrote them could not push to
that directory: GitHub refuses a push from an OAuth app without the `workflow`
scope, and the GitHub API path was equally unavailable. Everything else in the
branch pushed normally.

## To activate

Run these from a checkout whose credentials carry the `workflow` scope — an
ordinary `git push` from your own machine does; this session's did not.

```bash
git rm -q .github/workflows/ci.yml
git mv -f .github/workflows-pending/docs.yml    .github/workflows/docs.yml
git mv    .github/workflows-pending/quality.yml .github/workflows/quality.yml
git mv    .github/workflows-pending/release.yml .github/workflows/release.yml
git rm -q .github/workflows-pending/README.md
git commit -m "ci: activate the quality, docs, and release workflows"
git push
```

Two details in that order matter:

- **`docs.yml` needs `-f`**, because it overwrites the old Jupyter Book
  workflow of the same name.
- **Move `docs.yml` before removing the others.** `git rm`-ing both existing
  workflows first leaves `.github/workflows/` empty, git drops the now-empty
  directory, and the subsequent `git mv` fails with a confusing
  `No such file or directory` pointing at the *source* path.

The `git rm` of `ci.yml` matters too: `quality.yml` replaces it, and leaving
both in place means every push runs two overlapping test matrices.

## What each file does

**`quality.yml`** replaces `ci.yml`. The old workflow *counted* errors and passed
if `mypy` reported fewer than 50 and `ruff` fewer than 5000 — so it stayed green
while the codebase carried hundreds of both. The new one has two jobs:

- `quality`: `ruff format --check`, `ruff check`, `mypy`, and `bandit`, each
  required to be clean.
- `test`: Python 3.10–3.12 across Linux, macOS, and Windows, with a 70% coverage
  gate (was 46%) and a Codecov upload from the Linux/3.12 leg.

**`docs.yml`** builds the MkDocs site with `--strict` and deploys it to GitHub
Pages. It replaces a workflow that built a Jupyter Book and pushed the rendered
HTML into a second repository (`evotext/evotext.github.io`) using a deploy token.

Two things need setting up once, outside the repository:

1. **Pages**: repository Settings → Pages → Source → *GitHub Actions*.
2. **Custom domain**: `docs/CNAME` claims `row2vec.tresoldi.org`, so that name
   needs a DNS `CNAME` record pointing at `tresoldi.github.io`. Until then,
   delete `docs/CNAME` and drop `site_url` in `mkdocs.yml` to the default
   `https://tresoldi.github.io/row2vec/`.

**`release.yml`** is new. On a `v*` tag it builds the sdist and wheel, runs
`twine check`, and publishes to PyPI via **trusted publishing** (OIDC) — no API
token in secrets. This requires registering the repository as a trusted publisher
on PyPI (PyPI → your project → Publishing) and creating a `pypi` environment in
the repository settings.

`.github/dependabot.yml` pushed normally and is already in place.
