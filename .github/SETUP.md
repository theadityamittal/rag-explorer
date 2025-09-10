# GitHub CI/CD Setup Guide

This guide explains how to configure the GitHub repository secrets and settings needed for the CI/CD pipeline to work properly.

## Required GitHub Secrets

Navigate to your repository → Settings → Secrets and variables → Actions, then add these secrets:

### 🔑 API Keys

#### `GOOGLE_API_KEY` (Required)
- **Purpose**: Real Gemini API testing in CI/CD pipeline
- **How to get**: 
  1. Go to [Google AI Studio](https://aistudio.google.com/)
  2. Create an API key
  3. Copy the key (starts with `AIza...`)
- **Cost impact**: ~$0.05/month for CI testing (uses free tier)
- **Example**: `AIzaSyDh-example-key-here-1234567890`

### 📦 PyPI Publishing

#### `PYPI_API_TOKEN` (Required for releases)
- **Purpose**: Publishing to production PyPI
- **How to get**:
  1. Create account at [pypi.org](https://pypi.org/)
  2. Go to Account Settings → API tokens
  3. Create token with scope for your project
- **Format**: `pypi-AgE...` (starts with `pypi-`)

#### `TEST_PYPI_API_TOKEN` (Required for releases)
- **Purpose**: Testing releases on TestPyPI first
- **How to get**:
  1. Create account at [test.pypi.org](https://test.pypi.org/)
  2. Go to Account Settings → API tokens  
  3. Create token with scope for your project
- **Format**: `pypi-AgE...` (starts with `pypi-`)

### 🔒 Optional Secrets

#### `CODECOV_TOKEN` (Optional)
- **Purpose**: Code coverage reporting
- **How to get**: Create account at [codecov.io](https://codecov.io/) and link repository

## GitHub Repository Settings

### Branch Protection Rules

Set up branch protection for `main` branch:

1. Go to Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable:
   - [ ] Require status checks to pass before merging
   - [ ] Require branches to be up to date before merging
   - [ ] Status checks that are required:
     - `Code Quality`
     - `Package Build & CLI Test`
     - `Unit Tests (ubuntu-latest, 3.11)`
   - [ ] Require review from code owners
   - [ ] Restrict pushes that create files
   - [ ] Do not allow bypassing the above settings

### Environments (for PyPI Publishing)

Create protected environments for PyPI publishing:

#### TestPyPI Environment
1. Go to Settings → Environments → New environment
2. Name: `testpypi` 
3. Add environment secrets:
   - `TEST_PYPI_API_TOKEN`: Your TestPyPI token
4. Environment protection rules:
   - [ ] Required reviewers: Add yourself
   - [ ] Wait timer: 0 minutes
   - [ ] Deployment branches: Only `main`

#### PyPI Environment  
1. Go to Settings → Environments → New environment
2. Name: `pypi`
3. Add environment secrets:
   - `PYPI_API_TOKEN`: Your production PyPI token
4. Environment protection rules:
   - [ ] Required reviewers: Add yourself
   - [ ] Wait timer: 5 minutes (safety buffer)
   - [ ] Deployment branches: Only `main`

## Workflow Permissions

Ensure workflows have proper permissions:

1. Go to Settings → Actions → General
2. Under "Workflow permissions":
   - Select "Read and write permissions"
   - [ ] Allow GitHub Actions to create and approve pull requests

## Cost Controls

The CI/CD pipeline includes built-in cost controls:

### Gemini API Usage Limits
```yaml
# Built into workflows:
MONTHLY_BUDGET_USD: "2.0"      # $2 max per month
MAX_REQUESTS_PER_CI: "50"      # 50 API calls per CI run  
GOOGLE_MODEL: "gemini-1.5-flash"  # Cheapest model
```

### When Real API Tests Run
- ✅ **Main branch pushes**: Real Gemini API testing
- ❌ **Pull requests**: Mocked providers only (to save costs)
- ✅ **Releases**: Full real API testing
- ❌ **Feature branches**: Mocked providers only

## Testing the Setup

### 1. Test Basic CI Pipeline

Create a simple change and push to a feature branch:

```bash
git checkout -b test-ci
echo "# Test" >> test.md
git add test.md
git commit -m "test: verify CI pipeline"
git push origin test-ci
```

This should trigger:
- ✅ Code quality checks
- ✅ Unit tests (all platforms)
- ✅ Package build test
- ❌ Integration tests (skipped on feature branches)

### 2. Test API Integration

Push to main branch to test real Gemini API:

```bash
git checkout main
git merge test-ci
git push origin main
```

This should trigger:
- ✅ All CI checks
- ✅ Real Gemini API integration tests
- ✅ Cost tracking validation

### 3. Test Release Process

Create a test release:

```bash
git tag v0.1.1
git push origin v0.1.1
# Go to GitHub → Releases → Create release from tag
```

This should trigger:
- ✅ Release validation
- ✅ TestPyPI publishing  
- ✅ Installation testing
- ✅ Production PyPI publishing (if approved)

## Troubleshooting

### Common Issues

#### ❌ "Secret not found" errors
- **Solution**: Double-check secret names match exactly
- **Check**: Repository secrets are added (not organization level)

#### ❌ Gemini API tests failing  
- **Solution**: Verify `GOOGLE_API_KEY` is valid
- **Test**: Try API key manually with `curl`
- **Check**: Account has free quota available

#### ❌ PyPI publishing failing
- **Solution**: Verify API tokens are correct
- **Check**: Package name is available on PyPI
- **Test**: Try manual upload with `twine`

#### ❌ Tests timing out
- **Solution**: Check if external services are down
- **Temporary fix**: Re-run failed jobs

### Debug Commands

Test secrets locally (remove from file after testing):

```bash
# Test Gemini API
curl -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     -H "x-goog-api-key: YOUR_API_KEY" \
     https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent

# Test package build
python -m build
twine check dist/*

# Test CLI installation
pip install dist/*.whl
deflect-bot --version
```

## Security Best Practices

### Secret Management
- ✅ Never commit secrets to code
- ✅ Use environment-specific secrets
- ✅ Rotate API keys regularly
- ✅ Monitor secret usage in workflow logs

### API Key Security
- ✅ Use least-privilege API keys
- ✅ Set usage quotas/budgets
- ✅ Monitor for unusual usage patterns
- ✅ Revoke compromised keys immediately

### Cost Monitoring
- ✅ Set up billing alerts in provider accounts
- ✅ Monitor CI workflow costs monthly
- ✅ Adjust budget limits if needed
- ✅ Review API usage patterns

---

## Support

If you encounter issues with the CI/CD setup:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review workflow logs in the Actions tab
3. Create an issue with relevant logs and error messages
4. Tag with `ci/cd` label for faster response

**Remember**: The CI/CD pipeline is designed to be cost-effective and secure. Total monthly costs should be under $2 with normal usage.