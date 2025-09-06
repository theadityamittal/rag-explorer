# Configuration Management

## Environment Configuration

### Configuration Hierarchy

Configuration is loaded in the following order (later values override earlier ones):

1. Default values in code
2. Configuration files (`config.yaml`, `config.json`)
3. Environment variables
4. Command line arguments
5. Runtime overrides

### Environment Files

**Development (.env.development):**
```bash
# Application
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/myapp_dev
DATABASE_POOL_SIZE=5
DATABASE_TIMEOUT=30

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_TIMEOUT=5

# External APIs
API_BASE_URL=https://api-staging.example.com
API_TIMEOUT=30
API_RETRY_COUNT=3

# Features
FEATURE_NEW_DASHBOARD=true
FEATURE_BETA_API=false
```

**Production (.env.production):**
```bash
# Application
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:pass@db.prod.com:5432/myapp
DATABASE_POOL_SIZE=20
DATABASE_TIMEOUT=60

# Redis
REDIS_URL=redis://cache.prod.com:6379/0
REDIS_TIMEOUT=10

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# Monitoring
SENTRY_DSN=${SENTRY_DSN}
DATADOG_API_KEY=${DATADOG_API_KEY}

# External APIs
API_BASE_URL=https://api.example.com
API_TIMEOUT=60
API_RETRY_COUNT=5
```

### Configuration Loading

```python
import os
import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class DatabaseConfig:
    url: str = "sqlite:///app.db"
    pool_size: int = 5
    timeout: int = 30
    echo: bool = False

@dataclass
class RedisConfig:
    url: str = "redis://localhost:6379/0"
    timeout: int = 5
    max_connections: int = 10

@dataclass
class APIConfig:
    base_url: str = "https://api.example.com"
    timeout: int = 30
    retry_count: int = 3
    api_key: Optional[str] = None

@dataclass
class FeatureFlags:
    new_dashboard: bool = False
    beta_api: bool = False
    advanced_analytics: bool = False

@dataclass
class Config:
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "development"
    secret_key: str = "dev-secret-key"
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)

def load_config() -> Config:
    """Load configuration from multiple sources"""
    config = Config()
    
    # Load from YAML file if exists
    config_file = f"config/{os.getenv('ENVIRONMENT', 'development')}.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
            update_config_from_dict(config, file_config)
    
    # Override with environment variables
    update_config_from_env(config)
    
    return config

def update_config_from_env(config: Config):
    """Update config from environment variables"""
    config.debug = os.getenv('DEBUG', str(config.debug)).lower() == 'true'
    config.log_level = os.getenv('LOG_LEVEL', config.log_level)
    config.environment = os.getenv('ENVIRONMENT', config.environment)
    config.secret_key = os.getenv('SECRET_KEY', config.secret_key)
    
    # Database
    config.database.url = os.getenv('DATABASE_URL', config.database.url)
    config.database.pool_size = int(os.getenv('DATABASE_POOL_SIZE', config.database.pool_size))
    config.database.timeout = int(os.getenv('DATABASE_TIMEOUT', config.database.timeout))
    
    # Redis
    config.redis.url = os.getenv('REDIS_URL', config.redis.url)
    config.redis.timeout = int(os.getenv('REDIS_TIMEOUT', config.redis.timeout))
    
    # API
    config.api.base_url = os.getenv('API_BASE_URL', config.api.base_url)
    config.api.timeout = int(os.getenv('API_TIMEOUT', config.api.timeout))
    config.api.api_key = os.getenv('API_KEY', config.api.api_key)
    
    # Features
    config.features.new_dashboard = os.getenv('FEATURE_NEW_DASHBOARD', 'false').lower() == 'true'
    config.features.beta_api = os.getenv('FEATURE_BETA_API', 'false').lower() == 'true'

# Global config instance
config = load_config()
```

## Feature Flags

### Feature Flag Management

```python
from enum import Enum
from typing import Dict, Any
import redis
import json

class FeatureFlag(Enum):
    NEW_DASHBOARD = "new_dashboard"
    BETA_API = "beta_api"
    ADVANCED_ANALYTICS = "advanced_analytics"
    PAYMENT_V2 = "payment_v2"
    DARK_MODE = "dark_mode"

class FeatureFlagManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def is_enabled(self, flag: FeatureFlag, user_id: str = None, default: bool = False) -> bool:
        """Check if feature flag is enabled for user"""
        try:
            # Check cache first
            cache_key = f"feature:{flag.value}:{user_id or 'global'}"
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Check Redis
            flag_config = self._get_flag_config(flag)
            if not flag_config:
                return default
            
            # Global flag check
            if not flag_config.get('enabled', False):
                self.cache[cache_key] = False
                return False
            
            # User-specific checks
            if user_id and flag_config.get('user_whitelist'):
                result = user_id in flag_config['user_whitelist']
            elif user_id and flag_config.get('percentage_rollout'):
                result = self._is_in_percentage_rollout(user_id, flag_config['percentage_rollout'])
            else:
                result = flag_config.get('enabled', default)
            
            # Cache result
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logging.error(f"Error checking feature flag {flag.value}: {e}")
            return default
    
    def _get_flag_config(self, flag: FeatureFlag) -> Dict[str, Any]:
        """Get flag configuration from Redis"""
        try:
            config_json = self.redis.get(f"feature_flag:{flag.value}")
            if config_json:
                return json.loads(config_json)
        except Exception as e:
            logging.error(f"Error loading feature flag config: {e}")
        return {}
    
    def _is_in_percentage_rollout(self, user_id: str, percentage: int) -> bool:
        """Determine if user is in percentage rollout"""
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < percentage
    
    def set_flag(self, flag: FeatureFlag, enabled: bool = True, 
                 user_whitelist: List[str] = None, percentage_rollout: int = None):
        """Set feature flag configuration"""
        config = {
            'enabled': enabled,
            'user_whitelist': user_whitelist or [],
            'percentage_rollout': percentage_rollout
        }
        
        self.redis.setex(
            f"feature_flag:{flag.value}",
            86400,  # 24 hours TTL
            json.dumps(config)
        )
        
        # Clear cache
        self.cache.clear()

# Usage example
feature_flags = FeatureFlagManager(redis_client)

@app.route('/dashboard')
def dashboard():
    user_id = get_current_user_id()
    
    if feature_flags.is_enabled(FeatureFlag.NEW_DASHBOARD, user_id):
        return render_template('dashboard_v2.html')
    else:
        return render_template('dashboard_v1.html')
```

### Feature Flag API

```python
@app.route('/admin/feature-flags', methods=['GET'])
@require_admin
def list_feature_flags():
    """List all feature flags and their status"""
    flags = {}
    for flag in FeatureFlag:
        config = feature_flags._get_flag_config(flag)
        flags[flag.value] = {
            'enabled': config.get('enabled', False),
            'user_whitelist_count': len(config.get('user_whitelist', [])),
            'percentage_rollout': config.get('percentage_rollout')
        }
    return jsonify(flags)

@app.route('/admin/feature-flags/<flag_name>', methods=['POST'])
@require_admin
def update_feature_flag(flag_name):
    """Update feature flag configuration"""
    try:
        flag = FeatureFlag(flag_name)
    except ValueError:
        return jsonify({'error': 'Invalid flag name'}), 400
    
    data = request.get_json()
    feature_flags.set_flag(
        flag,
        enabled=data.get('enabled', False),
        user_whitelist=data.get('user_whitelist', []),
        percentage_rollout=data.get('percentage_rollout')
    )
    
    return jsonify({'status': 'updated'})
```

## Configuration Validation

### Schema Validation

```python
import cerberus
from typing import Dict, Any

CONFIG_SCHEMA = {
    'debug': {'type': 'boolean'},
    'log_level': {'type': 'string', 'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR']},
    'environment': {'type': 'string', 'allowed': ['development', 'staging', 'production']},
    'secret_key': {'type': 'string', 'minlength': 32},
    
    'database': {
        'type': 'dict',
        'schema': {
            'url': {'type': 'string', 'required': True},
            'pool_size': {'type': 'integer', 'min': 1, 'max': 100},
            'timeout': {'type': 'integer', 'min': 1}
        }
    },
    
    'redis': {
        'type': 'dict',
        'schema': {
            'url': {'type': 'string', 'required': True},
            'timeout': {'type': 'integer', 'min': 1},
            'max_connections': {'type': 'integer', 'min': 1}
        }
    },
    
    'api': {
        'type': 'dict',
        'schema': {
            'base_url': {'type': 'string', 'required': True},
            'timeout': {'type': 'integer', 'min': 1},
            'retry_count': {'type': 'integer', 'min': 0, 'max': 10}
        }
    }
}

def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary"""
    validator = cerberus.Validator(CONFIG_SCHEMA)
    
    if not validator.validate(config_dict):
        errors = validator.errors
        raise ValueError(f"Configuration validation failed: {errors}")
    
    return validator.normalized(config_dict)

# Custom validators
def validate_database_url(url: str) -> bool:
    """Validate database URL format"""
    import re
    pattern = r'^(postgresql|mysql|sqlite)://.+'
    return bool(re.match(pattern, url))

def validate_redis_url(url: str) -> bool:
    """Validate Redis URL format"""
    import re
    pattern = r'^redis://.*'
    return bool(re.match(pattern, url))
```

## Secret Management

### Environment-Based Secrets

```python
import os
import base64
from cryptography.fernet import Fernet

class SecretManager:
    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode())
    
    def get_secret(self, key: str, default: str = None, encrypted: bool = False) -> str:
        """Get secret from environment, optionally decrypt"""
        value = os.getenv(key, default)
        
        if encrypted and value and self.encryption_key:
            try:
                # Decrypt value
                decrypted = self.cipher.decrypt(base64.b64decode(value))
                return decrypted.decode()
            except Exception as e:
                logging.error(f"Failed to decrypt secret {key}: {e}")
                return default
        
        return value
    
    def set_secret(self, key: str, value: str, encrypted: bool = False) -> str:
        """Set secret, optionally encrypt"""
        if encrypted and self.encryption_key:
            try:
                # Encrypt value
                encrypted_value = self.cipher.encrypt(value.encode())
                return base64.b64encode(encrypted_value).decode()
            except Exception as e:
                logging.error(f"Failed to encrypt secret {key}: {e}")
                return value
        
        return value

# Usage
secrets = SecretManager()

# Regular secrets
database_password = secrets.get_secret('DATABASE_PASSWORD')
api_key = secrets.get_secret('API_KEY')

# Encrypted secrets
jwt_secret = secrets.get_secret('JWT_SECRET_ENCRYPTED', encrypted=True)
```

### AWS Secrets Manager Integration

```python
import boto3
import json
from botocore.exceptions import ClientError

class AWSSecretsManager:
    def __init__(self, region_name='us-east-1'):
        self.client = boto3.client('secretsmanager', region_name=region_name)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_secret(self, secret_name: str) -> Dict[str, str]:
        """Get secret from AWS Secrets Manager"""
        if secret_name in self.cache:
            return self.cache[secret_name]
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_value = json.loads(response['SecretString'])
            
            # Cache the result
            self.cache[secret_name] = secret_value
            return secret_value
            
        except ClientError as e:
            logging.error(f"Error retrieving secret {secret_name}: {e}")
            raise
    
    def get_secret_value(self, secret_name: str, key: str, default: str = None) -> str:
        """Get specific value from secret"""
        try:
            secret = self.get_secret(secret_name)
            return secret.get(key, default)
        except:
            return default

# Usage
aws_secrets = AWSSecretsManager()

# Get database credentials
db_credentials = aws_secrets.get_secret('myapp/database')
DATABASE_URL = f"postgresql://{db_credentials['username']}:{db_credentials['password']}@{db_credentials['host']}:5432/{db_credentials['database']}"

# Get specific API key
api_key = aws_secrets.get_secret_value('myapp/api-keys', 'stripe_api_key')
```

## Runtime Configuration Updates

### Hot Configuration Reload

```python
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloader(FileSystemEventHandler):
    def __init__(self, config_path: str, reload_callback):
        self.config_path = config_path
        self.reload_callback = reload_callback
        self.last_reload = time.time()
        
    def on_modified(self, event):
        if event.src_path == self.config_path:
            # Debounce multiple file system events
            now = time.time()
            if now - self.last_reload > 1.0:  # 1 second debounce
                self.last_reload = now
                try:
                    self.reload_callback()
                    logging.info(f"Configuration reloaded from {self.config_path}")
                except Exception as e:
                    logging.error(f"Error reloading configuration: {e}")

def setup_config_watcher(config_path: str, reload_callback):
    """Setup file watcher for configuration changes"""
    event_handler = ConfigReloader(config_path, reload_callback)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(config_path), recursive=False)
    observer.start()
    return observer

# Usage
def reload_config():
    global config
    config = load_config()
    # Notify other components about config change
    publish_config_change_event()

config_watcher = setup_config_watcher('config/production.yaml', reload_config)
```

### API for Configuration Updates

```python
@app.route('/admin/config', methods=['GET'])
@require_admin
def get_config():
    """Get current configuration (sanitized)"""
    sanitized_config = sanitize_config(config)
    return jsonify(sanitized_config)

@app.route('/admin/config', methods=['POST'])
@require_admin
def update_config():
    """Update configuration at runtime"""
    new_config = request.get_json()
    
    try:
        # Validate new configuration
        validate_config(new_config)
        
        # Update global config
        update_global_config(new_config)
        
        # Persist changes
        save_config_to_file(new_config)
        
        return jsonify({'status': 'updated'})
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

def sanitize_config(config: Config) -> Dict[str, Any]:
    """Remove sensitive values from config for API response"""
    config_dict = asdict(config)
    
    # Remove sensitive keys
    sensitive_keys = ['secret_key', 'api_key', 'password', 'token']
    
    def remove_sensitive(d):
        if isinstance(d, dict):
            return {k: remove_sensitive(v) if k.lower() not in sensitive_keys else '***' for k, v in d.items()}
        return d
    
    return remove_sensitive(config_dict)
```