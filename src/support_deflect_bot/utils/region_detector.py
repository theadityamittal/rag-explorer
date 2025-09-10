"""Regional compliance and detection utilities."""

import os
import logging
import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


# GDPR/EEA regions that require strict compliance
GDPR_REGIONS = {
    "EU",
    "EEA",
    "UK",
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
    "IS",
    "LI",
    "NO",
    "GB",
}

# Providers that are compliant in specific regions
PROVIDER_REGIONAL_COMPLIANCE = {
    "openai": ["global"],  # OpenAI is globally compliant
    "anthropic": ["global"],  # Claude is globally compliant
    "google_gemini_paid": ["global"],  # Google paid services are compliant
    "google_gemini_free": ["US", "CA", "AU", "JP"],  # Free tier restricted in EU
    "groq": ["US", "CA"],  # Groq primarily US-based
    "mistral": ["global"],  # Mistral is EU-based, globally compliant
    "test_provider": ["global"],  # Test provider for testing purposes
    "ollama": ["global"],  # Local Ollama is globally compliant
}


@dataclass
class RegionInfo:
    """Information about a user's region and compliance requirements."""

    code: str
    name: str
    requires_gdpr_compliance: bool
    allowed_providers: List[str]
    requires_consent: bool
    required_disclosures: List[str]


class RegionDetector:
    """Detect user region and determine compliance requirements."""

    def __init__(self):
        self._cache = {}
        self._region_info_cache = {}

    def detect_region(self) -> str:
        """Auto-detect user region using multiple methods.

        Returns:
            ISO country code or 'unknown' if detection fails
        """
        # Method 1: Check environment variable (highest priority)
        env_region = os.getenv("USER_REGION", "").upper()
        if env_region and env_region != "AUTO":
            logger.debug(f"Using region from environment: {env_region}")
            return env_region

        # Method 2: Try IP geolocation (with fallback)
        try:
            detected_region = self._detect_via_ip_geolocation()
            if detected_region:
                logger.debug(f"Detected region via IP: {detected_region}")
                return detected_region
        except Exception as e:
            logger.warning(f"IP geolocation failed: {e}")

        # Method 3: Check system locale as fallback
        try:
            locale_region = self._detect_via_locale()
            if locale_region:
                logger.debug(f"Using region from locale: {locale_region}")
                return locale_region
        except Exception as e:
            logger.warning(f"Locale detection failed: {e}")

        # Fallback: Unknown region (assume US for safety)
        logger.warning("Could not detect region, defaulting to US")
        return "US"

    def _detect_via_ip_geolocation(self) -> Optional[str]:
        """Detect region via IP geolocation service."""
        # Use httpbin.org for simple IP detection (no external dependencies)
        try:
            response = requests.get("http://httpbin.org/ip", timeout=3)
            if response.status_code == 200:
                ip_info = response.json()
                ip = ip_info.get("origin", "").split(",")[0].strip()

                if ip:
                    # Use a free IP geolocation service
                    geo_response = requests.get(
                        f"http://ip-api.com/json/{ip}", timeout=3
                    )
                    if geo_response.status_code == 200:
                        geo_data = geo_response.json()
                        country_code = geo_data.get("countryCode", "").upper()
                        if country_code:
                            return country_code

        except requests.RequestException:
            # Network issues, fall back to other methods
            pass

        return None

    def _detect_via_locale(self) -> Optional[str]:
        """Detect region via system locale."""
        import locale

        try:
            # Get the default locale
            default_locale = locale.getdefaultlocale()
            if default_locale and default_locale[0]:
                # Extract country code from locale (e.g., 'en_US' -> 'US')
                parts = default_locale[0].split("_")
                if len(parts) > 1:
                    return parts[1].upper()
        except Exception:
            pass

        return None

    def get_region_info(self, region: str) -> RegionInfo:
        """Get comprehensive information about a region.

        Args:
            region: ISO country code

        Returns:
            RegionInfo with compliance requirements
        """
        if region in self._region_info_cache:
            return self._region_info_cache[region]

        is_gdpr = self.is_gdpr_region(region)

        info = RegionInfo(
            code=region,
            name=self._get_region_name(region),
            requires_gdpr_compliance=is_gdpr,
            allowed_providers=self.get_compliant_providers(region),
            requires_consent=is_gdpr,
            required_disclosures=self.get_required_disclosures(region),
        )

        self._region_info_cache[region] = info
        return info

    def is_gdpr_region(self, region: str) -> bool:
        """Check if region requires GDPR compliance.

        Args:
            region: ISO country code

        Returns:
            True if GDPR compliance required
        """
        return region.upper() in GDPR_REGIONS

    def get_compliant_providers(self, region: str) -> List[str]:
        """Get list of legally compliant provider names for region.

        Args:
            region: ISO country code

        Returns:
            List of compliant provider names
        """
        compliant = []

        for provider, regions in PROVIDER_REGIONAL_COMPLIANCE.items():
            if "global" in regions or region.upper() in [r.upper() for r in regions]:
                compliant.append(provider)

        return compliant

    def get_required_disclosures(self, region: str) -> List[str]:
        """Get required legal disclosures for region.

        Args:
            region: ISO country code

        Returns:
            List of required disclosure statements
        """
        disclosures = [
            "This system uses AI to process your queries.",
            "AI responses may contain inaccuracies and should not be considered as professional advice.",
        ]

        if self.is_gdpr_region(region):
            disclosures.extend(
                [
                    "Your data is processed under GDPR protections.",
                    "You have the right to request data deletion.",
                    "You have the right to data portability.",
                    "You can withdraw consent at any time.",
                    "We process your data based on your consent or legitimate interest.",
                ]
            )

        return disclosures

    def _get_region_name(self, region: str) -> str:
        """Get human-readable region name."""
        region_names = {
            "US": "United States",
            "CA": "Canada",
            "GB": "United Kingdom",
            "UK": "United Kingdom",
            "DE": "Germany",
            "FR": "France",
            "IT": "Italy",
            "ES": "Spain",
            "NL": "Netherlands",
            "BE": "Belgium",
            "EU": "European Union",
            "EEA": "European Economic Area",
            "AU": "Australia",
            "JP": "Japan",
            "SG": "Singapore",
            "IN": "India",
            "BR": "Brazil",
            "MX": "Mexico",
        }

        return region_names.get(region.upper(), region.upper())


class ComplianceChecker:
    """Check legal compliance requirements for providers and regions."""

    def __init__(self, region_detector: Optional[RegionDetector] = None):
        self.region_detector = region_detector or RegionDetector()

    def check_provider_compliance(self, provider_name: str, region: str) -> bool:
        """Check if provider is compliant for given region.

        Args:
            provider_name: Name of the provider to check
            region: ISO country code

        Returns:
            True if provider is compliant for the region
        """
        compliant_providers = self.region_detector.get_compliant_providers(region)
        return provider_name in compliant_providers

    def require_consent(self, region: str) -> bool:
        """Check if user consent is required for region.

        Args:
            region: ISO country code

        Returns:
            True if explicit consent is required
        """
        return self.region_detector.is_gdpr_region(region)

    def validate_provider_for_region(
        self, provider_name: str, region: str
    ) -> Dict[str, Any]:
        """Validate provider compliance and return detailed status.

        Args:
            provider_name: Provider name to validate
            region: Target region

        Returns:
            Dict with compliance status and details
        """
        is_compliant = self.check_provider_compliance(provider_name, region)
        region_info = self.region_detector.get_region_info(region)

        return {
            "provider": provider_name,
            "region": region,
            "is_compliant": is_compliant,
            "requires_gdpr": region_info.requires_gdpr_compliance,
            "requires_consent": region_info.requires_consent,
            "allowed_providers": region_info.allowed_providers,
            "required_disclosures": region_info.required_disclosures,
            "reason": self._get_compliance_reason(provider_name, region, is_compliant),
        }

    def _get_compliance_reason(
        self, provider_name: str, region: str, is_compliant: bool
    ) -> str:
        """Get human-readable reason for compliance status."""
        if is_compliant:
            return f"{provider_name} is approved for use in {region}"

        if self.region_detector.is_gdpr_region(region):
            if "free" in provider_name.lower():
                return f"Free tier services are not permitted in GDPR regions like {region}"
            else:
                return f"{provider_name} does not meet GDPR compliance requirements for {region}"

        return f"{provider_name} is not available in {region}"


# Convenience functions for direct usage
def is_gdpr_region(region: str) -> bool:
    """Quick check if region requires GDPR compliance."""
    return region.upper() in GDPR_REGIONS


def detect_user_region() -> str:
    """Quick region detection."""
    detector = RegionDetector()
    return detector.detect_region()


def get_compliant_providers_for_region(region: str) -> List[str]:
    """Quick provider compliance check."""
    detector = RegionDetector()
    return detector.get_compliant_providers(region)
