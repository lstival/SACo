# SACo Website

This directory contains the Jekyll-based GitHub Pages website for the SACo (Semantically-Aware Contrastive Learning) research project.

## Local Development

1. **Install dependencies**
   ```powershell
   cd docs
   bundle install
   ```

2. **Serve locally**
   ```powershell
   bundle exec jekyll serve --baseurl ""
   ```
   
   The site will be available at `http://localhost:4000`

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. Make sure GitHub Pages is configured to use the `docs` folder as the source in your repository settings.

## Structure

- `index.md` - Home page with abstract and architecture
- `downloads.md` - Model weights and evaluation scripts
- `results.md` - Quantitative and qualitative results with BibTeX citation
- `_layouts/default.html` - Main layout template
- `assets/css/style.css` - Styling
- `assets/js/main.js` - JavaScript functionality
- `assets/images/` - All images and figures

## Features

- 🌓 Dark mode support
- 📱 Responsive design
- 📋 Copy-to-clipboard for BibTeX citations
- 🎨 Modern scientific aesthetic
- ⚡ Fast and lightweight
