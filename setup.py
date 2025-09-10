"""Setup script for Support Deflect Bot."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies (no LLM providers included)
base_requirements = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.6", 
    "pydantic==2.9.2",
    "python-dotenv==1.0.1",
    "httpx>=0.27.0",
    "requests>=2.32.3",
    "chromadb==0.5.5",
    "beautifulsoup4>=4.12.3",
    "lxml>=5.2.2",
    "click>=8.1.0",
    "rich>=13.0.0",
    "tabulate>=0.9.0",
]

# Provider-specific dependencies
api_providers = [
    "openai>=1.0.0",
    "groq>=0.4.0", 
    "mistralai>=0.1.0",
    "google-generativeai>=0.3.0",
    "anthropic>=0.25.0",
]

local_providers = [
    "ollama>=0.3.0",
]

dev_dependencies = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

setup(
    name="support-deflect-bot",
    version="0.1.0",
    author="Aditya Mittal",
    author_email="theadityamittal@gmail.com",
    description="Intelligent document Q&A with multi-provider LLM support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theadityamittal/support-deflect-bot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=base_requirements,
    extras_require={
        # Core API providers (cost-effective, legally compliant)
        "api": api_providers,
        # Local deployment providers  
        "local": local_providers,
        # All providers for complete functionality
        "all": api_providers + local_providers,
        # Development dependencies
        "dev": dev_dependencies,
        # Production deployment with API providers
        "prod": api_providers,
    },
    entry_points={
        "console_scripts": [
            "deflect-bot=src.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "support_deflect_bot": ["*.md", "*.txt", "*.json"],
    },
)