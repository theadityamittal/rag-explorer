# API Reference

## Authentication

All API requests require authentication using an API key in the header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/v1/endpoint
```

## Rate Limits

- **Free tier**: 100 requests per hour
- **Pro tier**: 1000 requests per hour  
- **Enterprise**: Unlimited requests

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Your rate limit ceiling for that given request
- `X-RateLimit-Remaining`: Number of requests left for the time window
- `X-RateLimit-Reset`: Time when the rate limit resets (Unix timestamp)

## Core Endpoints

### GET /api/v1/users
Retrieve user information.

**Parameters:**
- `limit` (optional): Maximum number of users to return (default: 10, max: 100)
- `offset` (optional): Number of users to skip (default: 0)

**Response:**
```json
{
  "data": [
    {
      "id": "user_123",
      "email": "user@example.com", 
      "created_at": "2023-01-15T10:30:00Z",
      "status": "active"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### POST /api/v1/users
Create a new user.

**Request Body:**
```json
{
  "email": "newuser@example.com",
  "password": "secure_password_123",
  "name": "John Doe"
}
```

**Response (201 Created):**
```json
{
  "id": "user_456",
  "email": "newuser@example.com",
  "name": "John Doe",
  "created_at": "2023-01-15T10:30:00Z",
  "status": "active"
}
```

### DELETE /api/v1/users/{id}
Delete a user by ID.

**Response (204 No Content):** Empty body

## Error Handling

All errors return JSON with the following structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Email is required",
    "details": {
      "field": "email",
      "reason": "missing_field"
    }
  }
}
```

**Common Error Codes:**
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Webhooks

Configure webhooks to receive real-time notifications about events.

**Setting up webhooks:**
1. Go to Settings > Webhooks in your dashboard
2. Add your endpoint URL (must be HTTPS)
3. Select events you want to receive
4. Save your webhook secret for signature validation

**Webhook payload example:**
```json
{
  "event": "user.created",
  "data": {
    "user_id": "user_789",
    "email": "webhook@example.com"
  },
  "timestamp": "2023-01-15T10:30:00Z"
}
```

**Verifying webhook signatures:**
```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```