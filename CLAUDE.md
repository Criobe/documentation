# Claude Code Guidelines for CRIOBE Documentation

This file contains specific instructions for Claude Code when working on this documentation project.

## Project Overview

Please refer to [README.md](README.md) for:
- Project structure and layout
- Development environment setup (Pixi or pip)
- Available commands for building and serving the documentation
- Deployment workflows

## Important: Documentation Only Repository

**This repository contains ONLY Markdown documentation files.**

- ❌ **DO NOT create Python scripts, shell scripts, or any executable code files**
- ❌ **DO NOT write automation scripts, utilities, or helper programs**
- ✅ **ONLY write and edit Markdown (`.md`) files in the `docs/` directory**
- ✅ **Code examples should exist ONLY within Markdown code blocks for documentation purposes**
- ✅ **Exception: GitHub Actions workflows in `.github/workflows/` for documentation deployment**

If the user requests code generation or scripting:
1. Politely clarify that this is a documentation-only repository
2. Suggest they navigate to the appropriate code repository (e.g., `coral_seg_yolo`, `grid_pose_detection`, etc.)
3. Offer to help write documentation about the code instead

**Allowed exceptions:**
- GitHub Actions workflow files (`.github/workflows/*.yml`) for building and deploying documentation
- Configuration files required by MkDocs (`mkdocs.yml`, `pixi.toml`, `requirements.txt`)

## Markdown Formatting Rules

### List Items with Trailing Spaces

When writing or editing Markdown lists, **always add two trailing spaces** at the end of each list item that continues with sub-content on the next line. This ensures proper rendering in MkDocs Material theme.

**Example - Correct formatting:**
```markdown
1. **Step 1: Enable GitHub Pages with GitHub Actions**

   Go to your repository settings and configure GitHub Pages.

2. **Step 2: Verify Workflow Permissions**

   Ensure proper permissions are set for the workflow.
```

**Example - Incorrect formatting:**
```markdown
1. **Step 1: Enable GitHub Pages with GitHub Actions**

   Go to your repository settings and configure GitHub Pages.
```

### When to Apply Trailing Spaces

- After numbered list items with sub-content
- After bulleted list items with sub-content
- After list items that are followed by indented paragraphs, code blocks, or nested lists
- Before blank lines within list items

### When NOT to Apply Trailing Spaces

- On the last line of a list item with no continuation
- On standalone list items with no sub-content
- Inside code blocks or code fences

## Documentation Standards

### File Organization

- Place new guides in the appropriate `docs/` subdirectory
- Follow the existing structure: `quickstart/`, `setup/`, `user-guide/`, `developer-guide/`
- Use lowercase with hyphens for file names: `my-new-guide.md`

### Code Blocks

- Always specify the language for syntax highlighting
- Use consistent indentation (4 spaces for nested blocks)
- Include comments where helpful

**Example:**
```markdown
```bash
# Navigate to project directory
cd ~/Projects/criobe_data

# Download test samples
wget https://storage.googleapis.com/data_criobe/test_samples.zip
```
```

### Admonitions

Use Material for MkDocs admonition syntax:

```markdown
!!! info "Title"
    Content here with proper indentation.

!!! warning "Important Notice"
    Warning content here.

!!! tip "Pro Tip"
    Helpful tip here.
```

### Links and References

- Use relative links for internal documentation: `[Guide](../setup/installation.md)`
- Use absolute URLs for external resources
- Include descriptive link text, avoid "click here"

## Testing Changes

Before committing documentation changes:

1. **Local preview:**
   ```bash
   pixi run serve
   # or
   mkdocs serve
   ```

2. **Build validation:**
   ```bash
   pixi run build-strict
   # or
   mkdocs build --strict
   ```

3. **Check for:**
   - Broken links
   - Missing images
   - Proper rendering of lists and admonitions
   - Code block syntax highlighting
   - Navigation structure

## Common Patterns

### File Paths

Use absolute paths from the user's home directory:
```bash
~/Projects/criobe_data/test_samples/
```

### Environment Variables

Reference `$DATA_ROOT` consistently:
```bash
export DATA_ROOT=~/Projects/criobe_data
cd $DATA_ROOT
```

### Command Examples

Show both the command and expected output when helpful:
```bash
# Verify installation
pixi --version
# Expected output: pixi 0.x.x
```

## Additional Notes

- This is a technical documentation site for ML/AI researchers and developers
- Maintain a professional, clear, and concise tone
- Prioritize accuracy over brevity
- Use diagrams (Mermaid) where they clarify complex workflows
- Keep the source of truth principle: authoritative guides in `setup/installation/for-developers/` should be referenced from quickstart guides
