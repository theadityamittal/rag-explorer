"""Claude Code provider implementation using user's Claude Pro subscription via subprocess."""

import logging
import subprocess
import time
import shutil
import tempfile
import os
from typing import Optional, Dict, Any
from pathlib import Path

from ..base import (
    LLMProvider,
    ProviderConfig,
    ProviderType,
    ProviderTier,
    ProviderError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class ClaudeCodeProvider(LLMProvider):
    """Claude Code subprocess provider using user's Claude Pro subscription."""

    def __init__(self, claude_path: Optional[str] = None, **kwargs):
        """Initialize Claude Code provider.

        Args:
            claude_path: Path to Claude Code executable
            **kwargs: Additional configuration options
        """
        super().__init__(api_key=None, **kwargs)  # No API key needed

        # Get Claude Code path from settings or parameter
        from ....utils.settings import CLAUDE_CODE_PATH

        self.claude_path = claude_path or CLAUDE_CODE_PATH

        # Find Claude Code executable
        self.executable_path = self._find_claude_code()

        logger.info(
            f"Initialized Claude Code provider with path: {self.executable_path}"
        )

    def get_config(self) -> ProviderConfig:
        """Get Claude Code provider configuration."""
        return ProviderConfig(
            name="Claude Code",
            provider_type=ProviderType.LLM,
            cost_per_million_tokens_input=0.0,  # Uses existing subscription
            cost_per_million_tokens_output=0.0,
            max_context_length=200000,  # Claude has large context window
            rate_limit_rpm=12,  # Pro plan limits (approx 45 messages / 5 hours)
            rate_limit_tpm=300000,  # Rough estimate
            supports_streaming=False,  # Subprocess doesn't support streaming
            requires_api_key=False,  # Uses existing subscription
            tier=ProviderTier.PREMIUM,  # Pro subscription tier
            regions_supported=["global"],  # Works everywhere Claude Pro works
            gdpr_compliant=True,  # Anthropic is GDPR compliant
            models_available=[
                "claude-3-sonnet",
                "claude-3-haiku",
                "claude-3.5-sonnet",
                "claude-3-opus",
            ],
        )

    def is_available(self) -> bool:
        """Check if Claude Code is available and accessible."""
        if not self.executable_path:
            return False

        try:
            # Test Claude Code with version check
            result = subprocess.run(
                [self.executable_path, "--version"],
                capture_output=True,
                timeout=10,
                text=True,
            )

            success = result.returncode == 0
            if success:
                logger.debug(f"Claude Code version: {result.stdout.strip()}")
            else:
                logger.debug(f"Claude Code version check failed: {result.stderr}")

            return success

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"Claude Code availability check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.executable_path:
                return {
                    "status": "unhealthy",
                    "error": "Claude Code executable not found",
                    "provider": "claude_code",
                }

            # Test basic functionality
            start_time = time.time()

            # Try a simple version check
            version_result = subprocess.run(
                [self.executable_path, "--version"],
                capture_output=True,
                timeout=10,
                text=True,
            )

            response_time = time.time() - start_time

            if version_result.returncode != 0:
                return {
                    "status": "unhealthy",
                    "error": f"Claude Code version check failed: {version_result.stderr}",
                    "provider": "claude_code",
                    "executable_path": self.executable_path,
                }

            # Try a minimal test prompt
            try:
                test_response = self._test_simple_prompt()
                test_success = bool(test_response and len(test_response) > 0)
            except Exception as e:
                test_success = False
                logger.debug(f"Claude Code test prompt failed: {e}")

            return {
                "status": "healthy" if test_success else "degraded",
                "response_time_ms": round(response_time * 1000, 2),
                "version": version_result.stdout.strip(),
                "executable_path": self.executable_path,
                "test_prompt_success": test_success,
                "provider": "claude_code",
                "timestamp": time.time(),
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "unhealthy",
                "error": "Claude Code health check timed out",
                "provider": "claude_code",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "claude_code",
                "timestamp": time.time(),
            }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate chat completion using Claude Code subprocess.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Model parameter (ignored - Claude Code uses default)
            temperature: Temperature parameter (ignored - Claude Code uses default)
            max_tokens: Max tokens parameter (ignored - Claude Code uses default)
            **kwargs: Additional parameters (ignored)

        Returns:
            Generated response text

        Raises:
            ProviderError: If subprocess call fails
            ProviderUnavailableError: If Claude Code not available
        """
        if not self.executable_path:
            raise ProviderUnavailableError(
                "Claude Code executable not found", provider="claude_code"
            )

        try:
            # Create a temporary file for the prompt
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as temp_file:
                # Format prompt for Claude Code
                full_prompt = f"{system_prompt}\n\n# User Query\n{user_prompt}\n\n# Assistant Response\n"
                temp_file.write(full_prompt)
                temp_file_path = temp_file.name

            try:
                # Execute Claude Code with the prompt file
                result = subprocess.run(
                    [
                        self.executable_path,
                        "--file",
                        temp_file_path,
                        "--mode",
                        "completion",
                    ],
                    capture_output=True,
                    timeout=120,
                    text=True,
                )  # 2 minute timeout

                if result.returncode != 0:
                    error_msg = (
                        result.stderr.strip() if result.stderr else "Unknown error"
                    )
                    raise ProviderError(
                        f"Claude Code failed: {error_msg}", provider="claude_code"
                    )

                response = result.stdout.strip()

                if not response:
                    raise ProviderError(
                        "Claude Code returned empty response", provider="claude_code"
                    )

                return response

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            raise ProviderError(
                "Claude Code timed out after 2 minutes", provider="claude_code"
            )
        except Exception as e:
            if isinstance(e, (ProviderError, ProviderUnavailableError)):
                raise
            raise ProviderError(
                f"Claude Code execution failed: {e}",
                provider="claude_code",
                original_error=e,
            )

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text. Claude Code doesn't provide direct token counting.

        Args:
            text: Text to count tokens for
            model: Model parameter (ignored)

        Returns:
            Estimated number of tokens
        """
        # Claude uses a similar tokenization to GPT models
        # Use base class estimation (1 token ≈ 4 characters)
        return self.estimate_tokens(text)

    def _find_claude_code(self) -> Optional[str]:
        """Find Claude Code executable in system PATH or common locations.

        Returns:
            Path to Claude Code executable or None if not found
        """
        # Try the configured path first
        if self.claude_path and self.claude_path != "claude":
            if Path(self.claude_path).exists():
                return self.claude_path

        # Common executable names
        executable_names = ["claude", "claude-code", "claude-cli"]

        # Try to find in PATH
        for name in executable_names:
            path = shutil.which(name)
            if path:
                logger.debug(f"Found Claude Code at: {path}")
                return path

        # Try common installation locations
        common_paths = [
            "/usr/local/bin/claude",
            "/usr/bin/claude",
            "~/.local/bin/claude",
            "/opt/claude/bin/claude",
            # Add more paths as needed
        ]

        for path_str in common_paths:
            path = Path(path_str).expanduser()
            if path.exists() and path.is_file():
                logger.debug(f"Found Claude Code at: {path}")
                return str(path)

        logger.warning("Claude Code executable not found in PATH or common locations")
        return None

    def _test_simple_prompt(self) -> Optional[str]:
        """Test Claude Code with a simple prompt.

        Returns:
            Response text or None if failed
        """
        try:
            simple_response = self.chat(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'Hello' to test the connection.",
                model=None,
            )
            return simple_response
        except Exception as e:
            logger.debug(f"Simple prompt test failed: {e}")
            return None

    def get_usage_info(self) -> Dict[str, Any]:
        """Get information about Claude Pro subscription usage limits.

        Returns:
            Usage information dictionary
        """
        return {
            "subscription_type": "Claude Pro",
            "monthly_cost": "$20 USD",
            "rate_limit_info": "Approximately 45 messages every 5 hours",
            "context_length": "200,000 tokens",
            "models_included": [
                "Claude 3 Sonnet",
                "Claude 3 Haiku",
                "Claude 3.5 Sonnet",
            ],
            "note": "Usage limits are shared with claude.ai web interface",
        }
