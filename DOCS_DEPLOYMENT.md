# Row2Vec Documentation Deployment Guide

This guide explains how to deploy Row2Vec documentation as a subpage of `evotext.github.io/row2vec/`.

## 🎯 Deployment Strategy

The documentation is built automatically and can be deployed to serve from `https://evotext.github.io/row2vec/` using the following approach:

### Architecture Overview

```
row2vec repository (private)
├── jupyter_book/          # Documentation source
├── .github/workflows/docs.yml  # Auto-builds docs
└── Generated artifacts    # Built HTML ready for deployment

evotext.github.io repository (public)
├── row2vec/              # Extracted documentation artifacts
├── index.html            # Main evotext landing page
└── Other projects...     # Other evotext projects
```

## 🚀 Automated Workflow

### Triggers
- **Automatic**: Commits to `main` branch that modify documentation
- **Manual**: GitHub Actions workflow dispatch

### Process
1. **Build**: Creates optimized HTML documentation
2. **Artifact**: Uploads build as downloadable artifact
3. **Integration**: Manual step to integrate with evotext.github.io

## 📋 Integration Steps

### For evotext.github.io Repository Maintainer

1. **Monitor Builds**
   - Watch for successful documentation builds in Actions tab
   - Download the latest `row2vec-docs-*` artifact

2. **Deploy to evotext.github.io**
   ```bash
   # In evotext.github.io repository
   cd evotext.github.io

   # Remove old documentation (if exists)
   rm -rf row2vec/

   # Extract new documentation
   unzip ~/Downloads/row2vec-docs-*.zip -d row2vec/

   # Commit changes
   git add row2vec/
   git commit -m "Update Row2Vec documentation"
   git push origin main
   ```

3. **Verify Deployment**
   - Visit `https://evotext.github.io/row2vec/`
   - Check that all links and navigation work correctly

## 🔧 Manual Build (Development)

For local testing and development:

```bash
cd jupyter_book
./build_docs.sh
# Open _build/html/index.html in browser
```

## 📁 File Structure

```
jupyter_book/
├── _config.yml              # Main config (for evotext subpage)
├── _config_subpage.yml      # Alternative config
├── _toc.yml                 # Table of contents
├── *.md                     # Documentation pages
├── build_docs.sh            # Local build script
└── _build/html/             # Generated documentation
```

## 🔍 Troubleshooting

### Build Failures
- Check GitHub Actions logs for detailed error messages
- Ensure all dependencies are properly installed
- Verify Jupyter Book configuration is valid

### Broken Links
- Ensure `baseurl` in `_config.yml` matches deployment URL
- Check that all internal links use relative paths
- Verify repository URLs are correct

### Styling Issues
- Clear browser cache after deployment
- Check that `.nojekyll` file is present in deployed directory
- Verify CSS/JS assets are correctly referenced

## 🎨 Customization

### Branding Integration
- Update `_config.yml` to match evotext branding
- Add custom CSS in `_static/` directory
- Modify footer to reference parent organization

### Navigation
- Edit `_toc.yml` to change page organization
- Use `_config.yml` HTML settings for navbar customization

## 📊 Best Practices

1. **Content Organization**
   - Keep documentation source in `jupyter_book/`
   - Use clear, descriptive file names
   - Maintain consistent markdown formatting

2. **Build Optimization**
   - Use `execute_notebooks: "off"` for faster builds
   - Minimize large assets and images
   - Enable appropriate caching

3. **Version Management**
   - Tag releases for stable documentation versions
   - Use semantic versioning for documentation updates
   - Maintain changelog for major documentation changes

## 🔗 Useful Links

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [evotext.github.io Repository](https://github.com/evotext/evotext.github.io)

## 🤝 Contributing

To contribute to documentation:

1. Edit files in `jupyter_book/` directory
2. Test locally with `./build_docs.sh`
3. Commit changes to trigger automatic build
4. Coordinate with evotext.github.io maintainer for deployment
