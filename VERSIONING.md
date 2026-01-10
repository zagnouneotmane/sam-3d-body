# Version Management with setuptools-scm

This project uses `setuptools-scm` for automatic version management from git tags.

## How It Works

1. **Version from Git Tags**: The version is automatically derived from git tags
2. **Fallback**: If no git tags exist, uses `0.1.0` (as specified in `pyproject.toml`)
3. **Auto-generation**: During build, setuptools-scm generates `sam_3d_body/_version.py`

## Creating a Release

### Step 1: Create a Git Tag

```bash
# Create an annotated tag
git tag -a v0.1.0 -m "Release version 0.1.0"

# Or create a lightweight tag
git tag v0.1.0
```

### Step 2: Push the Tag

```bash
git push origin v0.1.0
# Or push all tags
git push origin --tags
```

### Step 3: Install/Use

When someone installs from a specific tag:
```bash
pip install git+https://github.com/zagnouneotmane/sam-3d-body.git@v0.1.0
```

The version will automatically be set to `0.1.0` from the tag.

## Version Format

- Use semantic versioning: `MAJOR.MINOR.PATCH` (e.g., `1.0.0`, `0.2.1`)
- Tag format: `v0.1.0` or `0.1.0` (both work)
- The `v` prefix is optional but recommended

## Examples

```bash
# Tag a release
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0

# Tag a patch release
git tag -a v0.1.1 -m "Bug fixes"
git push origin v0.1.1

# Tag a minor release
git tag -a v0.2.0 -m "New features"
git push origin v0.2.0
```

## Checking Current Version

After installation:
```python
import sam_3d_body
print(sam_3d_body.__version__)
```

## Development

- During development (without tags), the fallback version `0.1.0` is used
- The `_version.py` file is generated during the build process
- Don't commit `_version.py` to git (add it to `.gitignore`)
