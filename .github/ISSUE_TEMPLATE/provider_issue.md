---
name: Provider Issue
about: Report an issue with a specific AI provider (OpenAI, Gemini, Ollama, etc.)
title: '[PROVIDER] '
labels: 'provider', 'bug'
assignees: 'theadityamittal'
---

# Provider Issue

## Provider Information

**Which provider is affected?**

- [ ] Google Gemini
- [ ] OpenAI (GPT-4, GPT-3.5)
- [ ] Groq
- [ ] Mistral
- [ ] Claude API (Anthropic)
- [ ] Claude Code
- [ ] Ollama (Local)
- [ ] Multiple providers: ___

**Provider Configuration:**

```bash
# Relevant configuration (remove sensitive info):
GOOGLE_API_KEY=***
DEFAULT_PROVIDER_STRATEGY=cost_optimized
# ... other relevant settings
```

## Issue Description

**What's the problem?**

A clear description of the provider-specific issue.

**Issue Type:**

- [ ] 🔌 Connection/Authentication issue
- [ ] 🚫 Rate limiting problems
- [ ] 💰 Cost/billing related
- [ ] 🔄 Fallback not working
- [ ] 🎯 Quality/accuracy issues
- [ ] ⚡ Performance problems
- [ ] 🔧 Configuration problems
- [ ] 🆕 New provider request
- [ ] Other: ___

## Reproduction Steps

1. Set provider configuration: `...`
2. Run command: `deflect-bot ...`
3. Expected provider to be used: `...`
4. See error/issue: `...`

## Provider Behavior

**Expected behavior:**
What should happen with this provider?

**Actual behavior:**
What actually happened?

**Fallback behavior:**
- [ ] Fallback to secondary provider worked
- [ ] Fallback to secondary provider failed
- [ ] No fallback attempted
- [ ] Not applicable

## Error Messages

**Provider-specific errors:**

```
Paste any provider-specific error messages here
```

**API Response (if applicable):**

```json
{
  "error": "paste API error response here"
}
```

## Provider Testing

**Testing done:**

- [ ] Tested API key validity directly
- [ ] Tested with different models
- [ ] Tested with different settings
- [ ] Tested provider health check: `deflect-bot ping`
- [ ] Tested fallback chain
- [ ] Checked provider status page

**Provider Status:**

- [ ] Provider API is operational (checked status page)
- [ ] API key is valid (tested separately)
- [ ] Account has sufficient credits/quota
- [ ] Rate limits not exceeded

## Cost/Usage Information

**For cost-related issues:**

- Monthly budget setting: $___
- Estimated usage: ___
- Cost alerts received: [Yes/No]
- Provider tier: [Free/Paid/Enterprise]

## Model Configuration

**Model settings:**

- LLM Model: [e.g., gemini-1.5-flash, gpt-4o-mini]
- Embedding Model: [e.g., text-embedding-004]
- Custom parameters: ___
- Temperature/settings: ___

## Regional/Compliance

**Geographic considerations:**

- User location: [Country/Region]
- GDPR compliance required: [Yes/No]
- Regional restrictions: ___

## Environment Details

**System information:**

- OS: ___
- Python version: ___
- Package version: ___
- Installation method: ___

**Network environment:**

- [ ] Corporate firewall
- [ ] VPN connection
- [ ] Proxy server
- [ ] Standard home/office network

## Additional Context

**Recent changes:**

- [ ] New API key
- [ ] Changed provider settings
- [ ] Updated package version
- [ ] Changed network environment
- [ ] Other: ___

**Related providers:**

- Does this affect other providers too? ___
- Fallback providers working? ___
- Provider selection strategy: ___

## Logs

**Provider-specific logs:**

```
Paste relevant logs showing provider interaction
```

**Cost tracking output:**

```bash
# Output of cost tracking (if enabled)
```

## Possible Solutions

**Have you tried:**

- [ ] Different API key
- [ ] Different model
- [ ] Different provider strategy
- [ ] Manual provider selection
- [ ] Clearing cache/database
- [ ] Reinstalling package

**Potential fixes you've considered:**
___

---

**For security: Please ensure no actual API keys, secrets, or sensitive data are included in this issue.**