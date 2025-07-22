# Building and Publishing the Pingala Shunya Package

This guide explains how to build and publish the Pingala Shunya PyPI package with multi-backend support.

## 📋 Prerequisites

### 1. Install Build Tools
```bash
pip install build twine
```

### 2. PyPI Account Setup
- Create account at [PyPI](https://pypi.org/account/register/)
- Create account at [TestPyPI](https://test.pypi.org/account/register/) (for testing)
- Generate API tokens for both accounts

### 3. Configure Authentication
Create `~/.pypirc`:
```ini
[pypi]
  username = __token__
  password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-YOUR_TEST_API_TOKEN_HERE
```

## 📦 Package Structure

Your package now has the following structure:

```
pingala-shunya/
├── pingala_shunya/              # Main package directory
│   ├── __init__.py             # Package initialization (v0.3.0)
│   ├── transcriber.py          # Multi-backend transcription
│   └── cli.py                  # Command-line interface
├── pyproject.toml              # Modern Python package configuration
├── README.md                   # Comprehensive documentation
├── LICENSE                     # MIT license
├── requirements.txt            # Core dependencies
├── MANIFEST.in                 # Files to include in distribution
├── example.py                  # Multi-backend examples
└── BUILD_AND_PUBLISH.md        # This file
```

## 🔨 Building the Package

### 1. Navigate to Package Directory
```bash
cd pingala-shunya
```

### 2. Clean Previous Builds (if any)
```bash
rm -rf dist/ build/ *.egg-info/
```

### 3. Verify Package Configuration
```bash
# Check pyproject.toml syntax
python -c "import tomllib; print('✅ pyproject.toml is valid')" 2>/dev/null || python -c "import tomli; print('✅ pyproject.toml is valid')"

# Verify package imports
python -c "from pingala_shunya import PingalaTranscriber; print('✅ Package imports successfully')"
```

### 4. Build the Package
```bash
python -m build
```

This creates files in the `dist/` directory:
- `pingala_shunya-0.3.0.tar.gz` (source distribution)
- `pingala_shunya-0.3.0-py3-none-any.whl` (wheel distribution)

### 5. Verify Build Contents
```bash
# List contents of wheel
python -m zipfile -l dist/pingala_shunya-0.3.0-py3-none-any.whl

# List contents of source distribution
tar -tzf dist/pingala_shunya-0.3.0.tar.gz
```

## 🧪 Testing Before Publishing

### 1. Test Installation from Local Build
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from wheel
pip install dist/pingala_shunya-0.3.0-py3-none-any.whl

# Test basic functionality
python -c "from pingala_shunya import PingalaTranscriber; print('✅ Installation successful')"

# Test CLI
pingala --version

# Cleanup
deactivate
rm -rf test_env
```

### 2. Publish to TestPyPI (Recommended)
```bash
# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pingala-shunya
```

## 🚀 Publishing to PyPI

### 1. Final Checks
- ✅ Version number is correct (0.3.0)
- ✅ All tests pass
- ✅ Documentation is complete
- ✅ License is included
- ✅ Contact information is correct (0@shunyalabs.ai)

### 2. Upload to PyPI
```bash
# Upload to production PyPI
python -m twine upload dist/*
```

### 3. Verify Publication
```bash
# Check package page
# Visit: https://pypi.org/project/pingala-shunya/

# Test installation
pip install pingala-shunya

# Test with different backends
pip install "pingala-shunya[all]"  # All backends
pip install "pingala-shunya[transformers]"  # Transformers only
```

## 📈 Post-Publication

### 1. Create Git Tag
```bash
git tag v0.3.0
git push origin v0.3.0
```

### 2. Update Documentation
- Update Shunya Labs website
- Add package to company portfolio
- Share on social media

### 3. Monitor and Maintain
- Watch for bug reports
- Monitor download statistics
- Plan future updates

## 🔄 Future Updates

For subsequent releases:

1. **Update Version**:
   - Update `pyproject.toml` version
   - Update `__init__.py` version
   - Update `cli.py` version

2. **Build and Test**:
   ```bash
   rm -rf dist/
   python -m build
   python -m twine upload --repository testpypi dist/*
   ```

3. **Publish**:
   ```bash
   python -m twine upload dist/*
   ```

## 🆘 Troubleshooting

### Common Issues:

**Build Fails:**
```bash
# Check dependencies
pip install --upgrade build setuptools wheel

# Verify Python version compatibility
python --version  # Should be 3.8+
```

**Upload Fails:**
```bash
# Update twine
pip install --upgrade twine

# Check credentials
python -m twine check dist/*
```

**Import Errors:**
```bash
# Check package structure
python -m pip show -f pingala-shunya

# Verify dependencies
pip install -r requirements.txt
```

### Version Conflicts:
If version already exists on PyPI:
1. Increment version in `pyproject.toml`, `__init__.py`, and `cli.py`
2. Rebuild package
3. Upload new version

## 📊 Package Information

- **Name**: pingala-shunya
- **Version**: 0.3.0
- **Author**: Shunya Labs
- **Email**: 0@shunyalabs.ai
- **Website**: https://shunyalabs.ai
- **Backends**: faster-whisper, transformers, openai-whisper
- **Python**: 3.8+

## 🎯 Success Checklist

Before publishing, ensure:

- [ ] Package builds successfully
- [ ] All backends tested
- [ ] CLI works correctly
- [ ] Documentation is complete
- [ ] Examples run without errors
- [ ] Version numbers are consistent
- [ ] Contact information is correct
- [ ] License is included
- [ ] TestPyPI upload successful

Ready to publish! 🚀 