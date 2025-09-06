# Security & Authentication

## Authentication Methods

### API Key Authentication

Generate and manage API keys in your dashboard:

```bash
# Using API key in requests
curl -H "Authorization: Bearer sk_live_abc123..." https://api.example.com/v1/endpoint
```

**Best practices:**
- Use different keys for different environments (dev, staging, prod)
- Rotate keys every 90 days
- Never commit keys to version control
- Use environment variables: `API_KEY=${API_KEY}`

### OAuth 2.0 Flow

**Authorization Code Flow:**

1. **Redirect user to authorization server:**
```
https://auth.example.com/oauth/authorize?
  response_type=code&
  client_id=your_client_id&
  redirect_uri=https://yourapp.com/callback&
  scope=read_user+write_data&
  state=random_state_string
```

2. **Exchange code for access token:**
```bash
curl -X POST https://auth.example.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=AUTHORIZATION_CODE" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "redirect_uri=https://yourapp.com/callback"
```

3. **Use access token:**
```bash
curl -H "Authorization: Bearer ACCESS_TOKEN" https://api.example.com/v1/user
```

### JWT Tokens

**Token structure:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

**Token validation in Python:**
```python
import jwt
from datetime import datetime, timedelta

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, 'your-secret-key', algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, 'your-secret-key', algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
```

## Password Security

### Password Requirements

Enforce strong passwords:
- Minimum 12 characters
- At least 1 uppercase letter
- At least 1 lowercase letter  
- At least 1 number
- At least 1 special character
- No common dictionary words
- No personal information (name, email, etc.)

### Password Hashing

**Always use bcrypt for password hashing:**

```python
import bcrypt

# Hash password
def hash_password(password):
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt)

# Verify password
def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)
```

**Configuration:**
- Use bcrypt with minimum 12 rounds (preferably 14-16)
- Never store plain text passwords
- Use secure random salts for each password

## Two-Factor Authentication

### TOTP (Time-based One-Time Password)

**Setup flow:**
1. Generate secret key for user
2. Display QR code with secret
3. User scans QR code with authenticator app
4. User enters verification code to confirm setup

```python
import pyotp
import qrcode

def generate_secret():
    return pyotp.random_base32()

def generate_qr_code(secret, user_email, service_name):
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user_email,
        issuer_name=service_name
    )
    qr = qrcode.make(totp_uri)
    return qr

def verify_totp(secret, token):
    totp = pyotp.TOTP(secret)
    return totp.verify(token, valid_window=1)
```

### SMS Backup Codes

Generate backup codes when 2FA is enabled:

```python
import secrets
import string

def generate_backup_codes(count=10):
    codes = []
    for _ in range(count):
        code = ''.join(secrets.choice(string.digits) for _ in range(8))
        codes.append(f"{code[:4]}-{code[4:]}")
    return codes
```

## Security Headers

### Essential HTTP Security Headers

```nginx
# Nginx configuration
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### Content Security Policy (CSP)

```html
<!-- Restrict script sources -->
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               script-src 'self' https://trusted-cdn.com; 
               style-src 'self' 'unsafe-inline';
               img-src 'self' data: https:;
               connect-src 'self' https://api.example.com;">
```

## Rate Limiting

### Implementation Patterns

**Token Bucket Algorithm:**
```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=100, window_size=3600):
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = defaultdict(list)
    
    def allow_request(self, identifier):
        now = time.time()
        # Remove old requests outside window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_size
        ]
        
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        return False
```

**Rate limiting by IP:**
- 100 requests per hour per IP for anonymous users
- 1000 requests per hour per IP for authenticated users
- Block IPs with suspicious patterns

## Encryption

### Data at Rest

```bash
# Database encryption
# Enable transparent data encryption (TDE) in PostgreSQL
echo "ssl = on" >> postgresql.conf
echo "ssl_cert_file = 'server.crt'" >> postgresql.conf  
echo "ssl_key_file = 'server.key'" >> postgresql.conf
```

### Data in Transit

**Always use HTTPS:**
```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
}
```

## Security Monitoring

### Failed Login Attempts

```python
# Monitor and block after failed attempts
def check_failed_attempts(ip_address, username):
    failed_attempts = get_failed_attempts(ip_address, username)
    if failed_attempts > 5:
        block_for_duration(ip_address, minutes=30)
        send_security_alert(username, ip_address)
```

### Audit Logging

Log security events:
- Login attempts (successful and failed)
- Password changes
- Permission changes
- API key usage
- Suspicious activities

```python
import logging

security_logger = logging.getLogger('security')
security_logger.info(f"Successful login: user={username} ip={ip_address}")
security_logger.warning(f"Failed login attempt: user={username} ip={ip_address}")
```