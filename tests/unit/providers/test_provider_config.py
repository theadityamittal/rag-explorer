"""
Unit tests for provider configuration system.

Tests provider configuration validation, loading, security,
and management across different provider implementations.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from tests.base import BaseProviderTest


class TestProviderConfigValidation(BaseProviderTest):
    """Test provider configuration validation."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_required_config_fields(self):
        """Test validation of required configuration fields."""
        # Arrange
        required_fields = ['name', 'api_key', 'model', 'endpoint']
        
        # Valid config
        valid_config = {
            'name': 'openai',
            'api_key': 'sk-test123',
            'model': 'gpt-4o-mini',
            'endpoint': 'https://api.openai.com/v1',
            'max_tokens': 4096
        }
        
        # Invalid configs (missing required fields)
        invalid_configs = [
            {'name': 'openai'},  # Missing api_key, model, endpoint
            {'api_key': 'sk-test123'},  # Missing name, model, endpoint
            {'name': 'openai', 'api_key': 'sk-test123'}  # Missing model, endpoint
        ]
        
        # Act & Assert
        def validate_config(config):
            return all(field in config for field in required_fields)
        
        assert validate_config(valid_config) is True
        
        for invalid_config in invalid_configs:
            assert validate_config(invalid_config) is False
            
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_type_validation(self):
        """Test validation of configuration field types."""
        # Arrange
        config = {
            'name': 'openai',
            'api_key': 'sk-test123',
            'model': 'gpt-4o-mini',
            'endpoint': 'https://api.openai.com/v1',
            'max_tokens': 4096,
            'temperature': 0.7,
            'enabled': True,
            'timeout': 30.0,
            'retry_attempts': 3,
            'headers': {'User-Agent': 'support-deflect-bot'}
        }
        
        # Type validation rules
        type_rules = {
            'name': str,
            'api_key': str,
            'model': str,
            'endpoint': str,
            'max_tokens': int,
            'temperature': (int, float),
            'enabled': bool,
            'timeout': (int, float),
            'retry_attempts': int,
            'headers': dict
        }
        
        # Act & Assert
        def validate_types(config, rules):
            for field, expected_type in rules.items():
                if field in config:
                    if not isinstance(config[field], expected_type):
                        return False, f"Field {field} should be {expected_type}"
            return True, "Valid"
        
        is_valid, message = validate_types(config, type_rules)
        assert is_valid is True
        assert message == "Valid"
        
        # Test invalid types
        invalid_config = config.copy()
        invalid_config['max_tokens'] = "4096"  # String instead of int
        
        is_valid, message = validate_types(invalid_config, type_rules)
        assert is_valid is False
        assert "max_tokens" in message
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_value_ranges(self):
        """Test validation of configuration value ranges."""
        # Arrange
        config = {
            'temperature': 0.7,
            'max_tokens': 4096,
            'timeout': 30.0,
            'retry_attempts': 3
        }
        
        # Value range rules
        range_rules = {
            'temperature': (0.0, 2.0),
            'max_tokens': (1, 32768),
            'timeout': (1.0, 300.0),
            'retry_attempts': (0, 10)
        }
        
        # Act & Assert
        def validate_ranges(config, rules):
            for field, (min_val, max_val) in rules.items():
                if field in config:
                    value = config[field]
                    if not (min_val <= value <= max_val):
                        return False, f"Field {field} should be between {min_val} and {max_val}"
            return True, "Valid"
        
        is_valid, message = validate_ranges(config, range_rules)
        assert is_valid is True
        
        # Test invalid ranges
        invalid_config = config.copy()
        invalid_config['temperature'] = 3.0  # Too high
        
        is_valid, message = validate_ranges(invalid_config, range_rules)
        assert is_valid is False
        assert "temperature" in message


class TestProviderConfigSecurity(BaseProviderTest):
    """Test provider configuration security features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_api_key_validation(self):
        """Test API key format validation and security."""
        # Arrange
        api_keys = {
            'openai': ['sk-test123', 'sk-proj-abc123', 'sk-1234567890'],
            'anthropic': ['sk-ant-api03-test123', 'sk-ant-api01-abc'],
            'groq': ['gsk_test123', 'gsk_abc456'],
            'invalid': ['', 'test', '123', 'invalid-key']
        }
        
        # API key validation patterns
        key_patterns = {
            'openai': r'^sk-[a-zA-Z0-9\-_]+$',
            'anthropic': r'^sk-ant-api\d+-[a-zA-Z0-9\-_]+$',
            'groq': r'^gsk_[a-zA-Z0-9\-_]+$'
        }
        
        # Act & Assert
        import re
        
        for provider, valid_keys in api_keys.items():
            if provider != 'invalid':
                pattern = key_patterns[provider]
                for key in valid_keys:
                    assert re.match(pattern, key) is not None, f"{provider} key {key} should be valid"
        
        # Test invalid keys
        for invalid_key in api_keys['invalid']:
            for provider, pattern in key_patterns.items():
                assert re.match(pattern, invalid_key) is None, f"Key {invalid_key} should be invalid"
                
    @pytest.mark.unit
    @pytest.mark.providers
    def test_api_key_masking(self):
        """Test API key masking in logs and output."""
        # Arrange
        api_key = "sk-test123456789abcdef"
        
        # Act
        def mask_api_key(key):
            if len(key) <= 8:
                return "*" * len(key)
            return key[:4] + "*" * (len(key) - 8) + key[-4:]
        
        masked_key = mask_api_key(api_key)
        
        # Assert
        assert masked_key.startswith("sk-t")
        assert masked_key.endswith("cdef")
        assert "*" in masked_key
        assert len(masked_key) == len(api_key)
        assert api_key != masked_key
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_sensitive_data_handling(self):
        """Test handling of sensitive configuration data."""
        # Arrange
        config = {
            'name': 'openai',
            'api_key': 'sk-secret123',
            'model': 'gpt-4o-mini',
            'endpoint': 'https://api.openai.com/v1',
            'organization_id': 'org-sensitive',
            'project_id': 'proj-sensitive'
        }
        
        sensitive_fields = ['api_key', 'organization_id', 'project_id']
        
        # Act
        def sanitize_config(config, sensitive_fields):
            sanitized = config.copy()
            for field in sensitive_fields:
                if field in sanitized:
                    value = sanitized[field]
                    if len(value) > 8:
                        sanitized[field] = value[:4] + "*" * (len(value) - 8) + value[-4:]
                    else:
                        sanitized[field] = "*" * len(value)
            return sanitized
        
        sanitized = sanitize_config(config, sensitive_fields)
        
        # Assert
        assert sanitized['name'] == 'openai'  # Non-sensitive unchanged
        assert sanitized['model'] == 'gpt-4o-mini'  # Non-sensitive unchanged
        assert sanitized['api_key'] != config['api_key']  # Sensitive masked
        assert "*" in sanitized['api_key']
        assert "*" in sanitized['organization_id']
        assert "*" in sanitized['project_id']


class TestProviderConfigLoading(BaseProviderTest):
    """Test provider configuration loading from various sources."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        # Arrange
        env_vars = {
            'OPENAI_API_KEY': 'sk-test123',
            'OPENAI_MODEL': 'gpt-4o-mini',
            'OPENAI_ENDPOINT': 'https://api.openai.com/v1',
            'OPENAI_MAX_TOKENS': '4096',
            'OPENAI_TEMPERATURE': '0.7'
        }
        
        # Act
        with patch.dict(os.environ, env_vars, clear=False):
            def load_config_from_env(provider_name):
                prefix = provider_name.upper()
                config = {}
                
                # Map environment variables to config
                env_mapping = {
                    f'{prefix}_API_KEY': 'api_key',
                    f'{prefix}_MODEL': 'model',
                    f'{prefix}_ENDPOINT': 'endpoint',
                    f'{prefix}_MAX_TOKENS': 'max_tokens',
                    f'{prefix}_TEMPERATURE': 'temperature'
                }
                
                for env_var, config_key in env_mapping.items():
                    if env_var in os.environ:
                        value = os.environ[env_var]
                        # Type conversion
                        if config_key in ['max_tokens']:
                            value = int(value)
                        elif config_key in ['temperature']:
                            value = float(value)
                        config[config_key] = value
                        
                return config
            
            config = load_config_from_env('openai')
        
        # Assert
        assert config['api_key'] == 'sk-test123'
        assert config['model'] == 'gpt-4o-mini'
        assert config['max_tokens'] == 4096
        assert config['temperature'] == 0.7
        assert isinstance(config['max_tokens'], int)
        assert isinstance(config['temperature'], float)
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_from_file(self):
        """Test loading configuration from file."""
        # Arrange
        config_data = {
            'providers': {
                'openai': {
                    'api_key': 'sk-file123',
                    'model': 'gpt-4o-mini',
                    'endpoint': 'https://api.openai.com/v1',
                    'max_tokens': 4096
                },
                'groq': {
                    'api_key': 'gsk_file456',
                    'model': 'llama-3.1-8b-instant',
                    'endpoint': 'https://api.groq.com/openai/v1'
                }
            }
        }
        
        # Act
        def load_config_from_dict(config_data, provider_name):
            return config_data.get('providers', {}).get(provider_name, {})
        
        openai_config = load_config_from_dict(config_data, 'openai')
        groq_config = load_config_from_dict(config_data, 'groq')
        missing_config = load_config_from_dict(config_data, 'missing')
        
        # Assert
        assert openai_config['api_key'] == 'sk-file123'
        assert openai_config['model'] == 'gpt-4o-mini'
        assert groq_config['api_key'] == 'gsk_file456'
        assert groq_config['model'] == 'llama-3.1-8b-instant'
        assert missing_config == {}
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_priority_override(self):
        """Test configuration priority: env vars > file > defaults."""
        # Arrange
        default_config = {
            'api_key': 'default-key',
            'model': 'default-model',
            'endpoint': 'https://default.com',
            'max_tokens': 1000,
            'temperature': 0.5
        }
        
        file_config = {
            'api_key': 'file-key',
            'model': 'file-model',
            'max_tokens': 2000
        }
        
        env_config = {
            'api_key': 'env-key',
            'temperature': 0.8
        }
        
        # Act - Merge configs with priority
        def merge_configs(default, file, env):
            config = default.copy()
            config.update(file)  # File overrides defaults
            config.update(env)   # Environment overrides file
            return config
        
        final_config = merge_configs(default_config, file_config, env_config)
        
        # Assert
        assert final_config['api_key'] == 'env-key'      # Env wins
        assert final_config['model'] == 'file-model'     # File wins over default
        assert final_config['endpoint'] == 'https://default.com'  # Default used
        assert final_config['max_tokens'] == 2000        # File wins over default
        assert final_config['temperature'] == 0.8        # Env wins


class TestProviderConfigManagement(BaseProviderTest):
    """Test provider configuration management operations."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_registration(self):
        """Test registering provider configurations."""
        # Arrange
        config_registry = {}
        
        configs = [
            {'name': 'openai', 'api_key': 'sk-test1', 'model': 'gpt-4o-mini'},
            {'name': 'groq', 'api_key': 'gsk_test2', 'model': 'llama-3.1-8b-instant'},
            {'name': 'ollama', 'endpoint': 'http://localhost:11434', 'model': 'llama2'}
        ]
        
        # Act
        def register_config(registry, config):
            name = config.get('name')
            if name:
                registry[name] = config
                return True
            return False
        
        for config in configs:
            success = register_config(config_registry, config)
            assert success is True
        
        # Assert
        assert len(config_registry) == 3
        assert 'openai' in config_registry
        assert 'groq' in config_registry
        assert 'ollama' in config_registry
        assert config_registry['openai']['model'] == 'gpt-4o-mini'
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_updates(self):
        """Test updating provider configurations."""
        # Arrange
        config_registry = {
            'openai': {
                'name': 'openai',
                'api_key': 'sk-old123',
                'model': 'gpt-3.5-turbo',
                'max_tokens': 1000
            }
        }
        
        updates = {
            'api_key': 'sk-new456',
            'model': 'gpt-4o-mini',
            'temperature': 0.7
        }
        
        # Act
        def update_config(registry, provider_name, updates):
            if provider_name in registry:
                registry[provider_name].update(updates)
                return True
            return False
        
        success = update_config(config_registry, 'openai', updates)
        
        # Assert
        assert success is True
        config = config_registry['openai']
        assert config['api_key'] == 'sk-new456'  # Updated
        assert config['model'] == 'gpt-4o-mini'  # Updated
        assert config['temperature'] == 0.7      # Added
        assert config['max_tokens'] == 1000      # Preserved
        assert config['name'] == 'openai'        # Preserved
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_validation_on_update(self):
        """Test configuration validation when updating."""
        # Arrange
        config_registry = {
            'openai': {
                'name': 'openai',
                'api_key': 'sk-test123',
                'model': 'gpt-4o-mini'
            }
        }
        
        valid_updates = {'temperature': 0.7, 'max_tokens': 4096}
        invalid_updates = {'temperature': 5.0, 'max_tokens': -100}  # Out of range
        
        # Act
        def validate_and_update(registry, provider_name, updates):
            # Validation rules
            if 'temperature' in updates:
                if not (0.0 <= updates['temperature'] <= 2.0):
                    return False, "Temperature must be between 0.0 and 2.0"
            if 'max_tokens' in updates:
                if updates['max_tokens'] < 1:
                    return False, "Max tokens must be positive"
            
            # Apply updates if valid
            if provider_name in registry:
                registry[provider_name].update(updates)
                return True, "Success"
            return False, "Provider not found"
        
        # Test valid updates
        success, message = validate_and_update(config_registry, 'openai', valid_updates)
        assert success is True
        assert config_registry['openai']['temperature'] == 0.7
        
        # Test invalid updates
        success, message = validate_and_update(config_registry, 'openai', invalid_updates)
        assert success is False
        assert "Temperature must be between" in message
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        # Arrange
        original_config = {
            'openai': {
                'name': 'openai',
                'api_key': 'sk-original',
                'model': 'gpt-4o-mini'
            },
            'groq': {
                'name': 'groq', 
                'api_key': 'gsk_original',
                'model': 'llama-3.1-8b-instant'
            }
        }
        
        # Act
        def backup_config(config):
            import copy
            return copy.deepcopy(config)
        
        def restore_config(backup):
            import copy
            return copy.deepcopy(backup)
        
        # Create backup
        backup = backup_config(original_config)
        
        # Modify original
        original_config['openai']['api_key'] = 'sk-modified'
        original_config['new_provider'] = {'name': 'new', 'api_key': 'test'}
        
        # Restore from backup
        restored_config = restore_config(backup)
        
        # Assert
        assert restored_config['openai']['api_key'] == 'sk-original'  # Restored
        assert 'new_provider' not in restored_config  # Modification not in backup
        assert len(restored_config) == 2  # Original structure preserved
        
        # Verify backup is independent
        backup['test'] = 'modification'
        assert 'test' not in restored_config  # Backup modification doesn't affect restore


class TestProviderConfigTemplates(BaseProviderTest):
    """Test provider configuration templates and presets."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_provider_config_templates(self):
        """Test provider configuration templates."""
        # Arrange
        templates = {
            'openai': {
                'name': 'openai',
                'endpoint': 'https://api.openai.com/v1',
                'model': 'gpt-4o-mini',
                'max_tokens': 4096,
                'temperature': 0.7,
                'headers': {'User-Agent': 'support-deflect-bot'},
                'retry_attempts': 3,
                'timeout': 30.0
            },
            'groq': {
                'name': 'groq',
                'endpoint': 'https://api.groq.com/openai/v1',
                'model': 'llama-3.1-8b-instant',
                'max_tokens': 8192,
                'temperature': 0.5,
                'retry_attempts': 2,
                'timeout': 15.0
            },
            'ollama': {
                'name': 'ollama',
                'endpoint': 'http://localhost:11434',
                'model': 'llama2',
                'max_tokens': 2048,
                'temperature': 0.8,
                'timeout': 60.0
            }
        }
        
        # Act
        def create_config_from_template(template_name, custom_values=None):
            if template_name not in templates:
                return None
            
            config = templates[template_name].copy()
            if custom_values:
                config.update(custom_values)
            return config
        
        # Test template usage
        openai_config = create_config_from_template('openai', {'api_key': 'sk-test123'})
        groq_config = create_config_from_template('groq', {'api_key': 'gsk_test456', 'temperature': 0.3})
        
        # Assert
        assert openai_config['name'] == 'openai'
        assert openai_config['api_key'] == 'sk-test123'  # Custom value
        assert openai_config['model'] == 'gpt-4o-mini'   # Template value
        
        assert groq_config['name'] == 'groq'
        assert groq_config['temperature'] == 0.3         # Custom override
        assert groq_config['model'] == 'llama-3.1-8b-instant'  # Template value
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_config_presets(self):
        """Test configuration presets for different use cases."""
        # Arrange
        presets = {
            'fast': {
                'temperature': 0.3,
                'max_tokens': 1000,
                'timeout': 10.0
            },
            'balanced': {
                'temperature': 0.7,
                'max_tokens': 4096,
                'timeout': 30.0
            },
            'creative': {
                'temperature': 1.2,
                'max_tokens': 8192,
                'timeout': 60.0
            }
        }
        
        base_config = {
            'name': 'openai',
            'api_key': 'sk-test123',
            'model': 'gpt-4o-mini',
            'endpoint': 'https://api.openai.com/v1'
        }
        
        # Act
        def apply_preset(config, preset_name):
            if preset_name not in presets:
                return config
            
            result = config.copy()
            result.update(presets[preset_name])
            return result
        
        fast_config = apply_preset(base_config, 'fast')
        creative_config = apply_preset(base_config, 'creative')
        
        # Assert
        assert fast_config['temperature'] == 0.3
        assert fast_config['max_tokens'] == 1000
        assert fast_config['timeout'] == 10.0
        assert fast_config['name'] == 'openai'  # Base preserved
        
        assert creative_config['temperature'] == 1.2
        assert creative_config['max_tokens'] == 8192
        assert creative_config['timeout'] == 60.0
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_dynamic_config_generation(self):
        """Test dynamic configuration generation based on requirements."""
        # Arrange
        requirements = {
            'use_case': 'embedding',
            'budget': 'low',
            'speed': 'fast',
            'local_only': False
        }
        
        # Act
        def generate_config(requirements):
            config = {}
            
            # Provider selection based on use case
            if requirements.get('use_case') == 'embedding':
                if requirements.get('local_only'):
                    config['name'] = 'ollama'
                    config['endpoint'] = 'http://localhost:11434'
                elif requirements.get('budget') == 'low':
                    config['name'] = 'groq'
                    config['endpoint'] = 'https://api.groq.com/openai/v1'
                else:
                    config['name'] = 'openai'
                    config['endpoint'] = 'https://api.openai.com/v1'
            
            # Speed configuration
            if requirements.get('speed') == 'fast':
                config['timeout'] = 10.0
                config['max_tokens'] = 1000
            else:
                config['timeout'] = 30.0
                config['max_tokens'] = 4096
            
            return config
        
        config = generate_config(requirements)
        
        # Assert
        assert config['name'] == 'groq'  # Low budget, not local_only
        assert config['timeout'] == 10.0  # Fast speed
        assert config['max_tokens'] == 1000  # Fast speed
        
        # Test local-only requirement
        local_requirements = requirements.copy()
        local_requirements['local_only'] = True
        
        local_config = generate_config(local_requirements)
        assert local_config['name'] == 'ollama'
        assert 'localhost' in local_config['endpoint']