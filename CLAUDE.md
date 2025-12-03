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

### List Formatting for MkDocs

MkDocs uses Python Markdown, which has specific requirements for list formatting. Follow these rules to ensure proper rendering:

#### 1. Four-Space Indentation Rule

**CRITICAL: All nested content within list items MUST be indented with 4 spaces (or one tab).**

This includes:
- Continuation paragraphs
- Code blocks
- Nested lists
- Admonitions
- Any other block-level content

**Example - Correct formatting:**
```markdown
1. **Step 1: Enable GitHub Pages with GitHub Actions**

    Go to your repository settings and configure GitHub Pages.

    This is a second paragraph, also indented 4 spaces.

2. **Step 2: Verify Workflow Permissions**

    Ensure proper permissions are set for the workflow.

    - Nested bullet point (4 spaces from list margin)
    - Another nested item
```

**Example - Incorrect formatting (2 or 3 spaces will break rendering):**
```markdown
1. **Step 1: Enable GitHub Pages with GitHub Actions**

  Only 2 spaces - this will NOT render correctly in MkDocs!
```

#### 2. Code Blocks Within Lists

When including code blocks inside list items:

- Add a blank line before and after the code block
- Indent the entire code block **8 spaces** (4 for the list item + 4 for the code block)

**Example:**
```markdown
1. **Install dependencies:**

    First, navigate to the project directory.

    ```bash
    cd ~/Projects/criobe_data
    pip install -r requirements.txt
    ```

    The installation should complete without errors.
```

#### 3. Trailing Spaces (Use Sparingly)

Two trailing spaces at the end of a line create a hard line break. However:

- **Avoid trailing spaces when possible** - they're invisible and many editors remove them
- Only use when you specifically need a line break without starting a new paragraph
- For most cases, use proper 4-space indentation instead

**When trailing spaces might be useful:**
```markdown
- Line one with two spaces after it··
  This continues on the next line (same paragraph, hard break)
```

**Better alternative (using proper indentation):**
```markdown
- Line one

    This is a new paragraph within the same list item (4 spaces).
```

#### 4. Blank Lines Before Lists

**CRITICAL: Always add a blank line before starting a list.**

Lists must be preceded by a blank line to render correctly in MkDocs. This applies to:
- Lists following paragraphs
- Lists following bold headers (e.g., `**Header**:`)
- Lists following any other block-level content

**Example - Correct formatting:**
```markdown
**Benefits of Centralized Data**:

- Download test data once, accessible by all modules
- Share CVAT images and annotations across modules
- No duplication of large datasets
```

**Example - Incorrect formatting:**
```markdown
**Benefits of Centralized Data**:
- Download test data once, accessible by all modules
- Share CVAT images and annotations across modules

This will render all items on one line without breaks!
```

#### 5. Blank Lines Between List Items

- Use blank lines between list items when they contain multiple paragraphs or blocks
- Blank lines help improve readability and ensure proper rendering
- Nested content after a blank line must still be indented 4 spaces

### Common List Formatting Mistakes

❌ **Wrong - No blank line before list:**
```markdown
**Header**:
- Item 1
- Item 2

Result: All items render on one line!
```

❌ **Wrong - 2 or 3 space indentation:**
```markdown
1. Item
  Only 2 spaces - breaks in MkDocs
```

❌ **Wrong - No indentation for continuation:**
```markdown
1. Item
Next line without indentation - breaks the list
```

❌ **Wrong - Insufficient code block indentation:**
```markdown
1. Item
    ```bash
    code here
    ```
Only 4 spaces total - code block won't render in list
```

✅ **Correct - Blank line before list + 4 space indentation:**
```markdown
**Header**:

- Item 1
- Item 2

1. Numbered item

    Continuation paragraph with 4 spaces.

    ```bash
    # Code block with 4 additional spaces (8 total)
    echo "hello"
    ```
```

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
