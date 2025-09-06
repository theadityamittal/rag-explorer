# Integration Guides

## Payment Gateway Integration

### Stripe Integration

**Setup and Configuration:**
```python
import stripe
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

class StripePaymentProcessor:
    def __init__(self):
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    
    def create_payment_intent(self, amount, currency='usd', customer_id=None):
        """Create a payment intent"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Amount in cents
                currency=currency,
                customer=customer_id,
                metadata={
                    'order_id': request.json.get('order_id'),
                    'user_id': request.json.get('user_id')
                }
            )
            return {
                'client_secret': intent.client_secret,
                'payment_intent_id': intent.id
            }
        except stripe.error.StripeError as e:
            raise PaymentError(f"Stripe error: {e}")
    
    def confirm_payment(self, payment_intent_id):
        """Confirm a payment intent"""
        try:
            intent = stripe.PaymentIntent.confirm(payment_intent_id)
            return intent.status == 'succeeded'
        except stripe.error.StripeError as e:
            raise PaymentError(f"Payment confirmation failed: {e}")
    
    def create_customer(self, email, name=None):
        """Create a Stripe customer"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name
            )
            return customer.id
        except stripe.error.StripeError as e:
            raise PaymentError(f"Customer creation failed: {e}")

@app.route('/payments/create-intent', methods=['POST'])
def create_payment_intent():
    """Create payment intent endpoint"""
    try:
        data = request.get_json()
        amount = data.get('amount')
        currency = data.get('currency', 'usd')
        customer_id = data.get('customer_id')
        
        processor = StripePaymentProcessor()
        result = processor.create_payment_intent(amount, currency, customer_id)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/webhooks/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks"""
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, processor.webhook_secret
        )
    except ValueError:
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError:
        return 'Invalid signature', 400
    
    # Handle the event
    if event['type'] == 'payment_intent.succeeded':
        handle_payment_success(event['data']['object'])
    elif event['type'] == 'payment_intent.payment_failed':
        handle_payment_failure(event['data']['object'])
    
    return jsonify({'status': 'success'})
```

### PayPal Integration

```python
import paypalrestsdk
from paypalrestsdk import Payment

paypalrestsdk.configure({
    "mode": "sandbox",  # or "live" for production
    "client_id": os.getenv('PAYPAL_CLIENT_ID'),
    "client_secret": os.getenv('PAYPAL_CLIENT_SECRET')
})

class PayPalPaymentProcessor:
    def create_payment(self, amount, currency='USD', return_url=None, cancel_url=None):
        """Create PayPal payment"""
        payment = Payment({
            "intent": "sale",
            "payer": {"payment_method": "paypal"},
            "redirect_urls": {
                "return_url": return_url or "http://localhost:8000/payment/return",
                "cancel_url": cancel_url or "http://localhost:8000/payment/cancel"
            },
            "transactions": [{
                "item_list": {
                    "items": [{
                        "name": "Payment",
                        "sku": "payment",
                        "price": str(amount),
                        "currency": currency,
                        "quantity": 1
                    }]
                },
                "amount": {
                    "total": str(amount),
                    "currency": currency
                },
                "description": "Payment description"
            }]
        })
        
        if payment.create():
            # Find approval URL
            for link in payment.links:
                if link.rel == "approval_url":
                    return {
                        'payment_id': payment.id,
                        'approval_url': link.href
                    }
        else:
            raise PaymentError(f"PayPal error: {payment.error}")
    
    def execute_payment(self, payment_id, payer_id):
        """Execute approved PayPal payment"""
        payment = Payment.find(payment_id)
        
        if payment.execute({"payer_id": payer_id}):
            return payment.state == 'approved'
        else:
            raise PaymentError(f"Payment execution failed: {payment.error}")
```

## Email Service Integration

### SendGrid Integration

```python
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

class EmailService:
    def __init__(self, api_key):
        self.sg = sendgrid.SendGridAPIClient(api_key=api_key)
    
    def send_email(self, to_email, subject, html_content, from_email=None, template_id=None, template_data=None):
        """Send email using SendGrid"""
        from_email = from_email or Email("noreply@yourapp.com")
        to_email = To(to_email)
        
        if template_id and template_data:
            # Use template
            mail = Mail(
                from_email=from_email,
                to_emails=to_email
            )
            mail.template_id = template_id
            mail.dynamic_template_data = template_data
        else:
            # Use HTML content
            content = Content("text/html", html_content)
            mail = Mail(from_email, to_email, subject, content)
        
        try:
            response = self.sg.send(mail)
            return response.status_code == 202
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False
    
    def send_bulk_email(self, emails, subject, template_id, template_data_list):
        """Send bulk emails with personalization"""
        mail = Mail()
        mail.from_email = Email("noreply@yourapp.com")
        mail.template_id = template_id
        mail.subject = subject
        
        # Add personalizations
        for email, template_data in zip(emails, template_data_list):
            personalization = Personalization()
            personalization.add_to(Email(email))
            personalization.dynamic_template_data = template_data
            mail.add_personalization(personalization)
        
        try:
            response = self.sg.send(mail)
            return response.status_code == 202
        except Exception as e:
            print(f"Bulk email sending failed: {e}")
            return False

# Usage
email_service = EmailService(os.getenv('SENDGRID_API_KEY'))

# Send welcome email
email_service.send_email(
    to_email='user@example.com',
    subject='Welcome!',
    template_id='d-1234567890',
    template_data={
        'first_name': 'John',
        'activation_link': 'https://yourapp.com/activate/token123'
    }
)
```

### AWS SES Integration

```python
import boto3
from botocore.exceptions import ClientError

class AWSEmailService:
    def __init__(self, region='us-east-1'):
        self.client = boto3.client('ses', region_name=region)
    
    def send_email(self, to_emails, subject, html_body, text_body=None, from_email=None):
        """Send email using AWS SES"""
        from_email = from_email or 'noreply@yourapp.com'
        
        try:
            response = self.client.send_email(
                Destination={
                    'ToAddresses': to_emails if isinstance(to_emails, list) else [to_emails]
                },
                Message={
                    'Body': {
                        'Html': {'Charset': 'UTF-8', 'Data': html_body},
                        'Text': {'Charset': 'UTF-8', 'Data': text_body or ''}
                    },
                    'Subject': {'Charset': 'UTF-8', 'Data': subject}
                },
                Source=from_email
            )
            return response['MessageId']
        except ClientError as e:
            print(f"Email sending failed: {e}")
            return None
    
    def send_templated_email(self, to_emails, template_name, template_data, from_email=None):
        """Send email using SES template"""
        from_email = from_email or 'noreply@yourapp.com'
        
        try:
            response = self.client.send_templated_email(
                Source=from_email,
                Destination={
                    'ToAddresses': to_emails if isinstance(to_emails, list) else [to_emails]
                },
                Template=template_name,
                TemplateData=json.dumps(template_data)
            )
            return response['MessageId']
        except ClientError as e:
            print(f"Templated email sending failed: {e}")
            return None
```

## Third-Party API Integration

### REST API Client with Retry Logic

```python
import requests
import time
import json
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class APIClient:
    def __init__(self, base_url, api_key=None, timeout=30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Configure session with retry strategy
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1  # Wait 1, 2, 4 seconds between retries
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with error handling"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text
                
        except requests.exceptions.Timeout:
            raise APIError(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise APIError("Connection error")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Handle rate limiting
                retry_after = e.response.headers.get('Retry-After', '60')
                raise APIError(f"Rate limited. Retry after {retry_after} seconds")
            else:
                raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
    
    def get(self, endpoint, params=None):
        """GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint, data=None, json_data=None):
        """POST request"""
        return self._make_request('POST', endpoint, data=data, json=json_data)
    
    def put(self, endpoint, data=None, json_data=None):
        """PUT request"""
        return self._make_request('PUT', endpoint, data=data, json=json_data)
    
    def delete(self, endpoint):
        """DELETE request"""
        return self._make_request('DELETE', endpoint)

# Usage example
class SlackIntegration:
    def __init__(self, webhook_url, bot_token=None):
        self.webhook_url = webhook_url
        self.client = APIClient('https://slack.com/api', bot_token)
    
    def send_message(self, channel, text, blocks=None):
        """Send message to Slack channel"""
        payload = {
            'channel': channel,
            'text': text
        }
        
        if blocks:
            payload['blocks'] = blocks
        
        return self.client.post('chat.postMessage', json_data=payload)
    
    def send_webhook_message(self, text, attachments=None):
        """Send message via webhook"""
        payload = {'text': text}
        if attachments:
            payload['attachments'] = attachments
        
        response = requests.post(self.webhook_url, json=payload)
        return response.status_code == 200

# Example usage
slack = SlackIntegration(
    webhook_url=os.getenv('SLACK_WEBHOOK_URL'),
    bot_token=os.getenv('SLACK_BOT_TOKEN')
)

# Send alert message
slack.send_webhook_message(
    text="🚨 High error rate detected!",
    attachments=[{
        'color': 'danger',
        'fields': [
            {'title': 'Error Rate', 'value': '15.2%', 'short': True},
            {'title': 'Time', 'value': '2023-01-15 10:30:00', 'short': True}
        ]
    }]
)
```

## Database Migrations

### Alembic (SQLAlchemy) Integration

```python
# alembic/env.py
from alembic import context
from sqlalchemy import engine_from_config, pool
from logging.config import fileConfig
import os
import sys

# Add your project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from myapp.models import Base
from myapp.config import DATABASE_URL

# Alembic Config object
config = context.config

# Set database URL from environment
config.set_main_option('sqlalchemy.url', DATABASE_URL)

# Configure logging
fileConfig(config.config_file_name)

# Target metadata for autogenerate support
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"}
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Migration Commands:**
```bash
# Initialize Alembic
alembic init alembic

# Generate migration from model changes
alembic revision --autogenerate -m "Add user table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history
```

**Custom Migration Script:**
```python
# migrations/versions/001_add_user_table.py
"""Add user table

Revision ID: 001
Revises: 
Create Date: 2023-01-15 10:30:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Create user table"""
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    
    # Create indexes
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_active', 'users', ['is_active'])

def downgrade():
    """Drop user table"""
    op.drop_index('idx_users_active')
    op.drop_index('idx_users_email')
    op.drop_table('users')
```

## Message Queue Integration

### Celery with Redis

```python
from celery import Celery
from kombu import Queue
import os

# Celery configuration
app = Celery('myapp')

app.conf.update(
    broker_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'myapp.tasks.send_email': {'queue': 'email'},
        'myapp.tasks.process_image': {'queue': 'media'},
        'myapp.tasks.generate_report': {'queue': 'reports'},
    },
    
    # Queue definitions
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('email', routing_key='email'),
        Queue('media', routing_key='media'),
        Queue('reports', routing_key='reports'),
    ),
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# Task definitions
@app.task(bind=True, max_retries=3)
def send_email(self, to_email, subject, template_id, template_data):
    """Send email task with retry logic"""
    try:
        email_service = EmailService()
        result = email_service.send_email(
            to_email=to_email,
            subject=subject,
            template_id=template_id,
            template_data=template_data
        )
        
        if not result:
            raise EmailError("Failed to send email")
        
        return f"Email sent successfully to {to_email}"
        
    except Exception as exc:
        print(f"Email task failed: {exc}")
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            retry_delay = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s
            raise self.retry(exc=exc, countdown=retry_delay)
        else:
            # Final failure - could send to dead letter queue
            return f"Email failed permanently: {exc}"

@app.task
def process_image(image_path, transformations):
    """Process image with transformations"""
    try:
        from PIL import Image
        
        img = Image.open(image_path)
        
        for transform in transformations:
            if transform['type'] == 'resize':
                img = img.resize(transform['size'])
            elif transform['type'] == 'rotate':
                img = img.rotate(transform['angle'])
        
        # Save processed image
        output_path = image_path.replace('.jpg', '_processed.jpg')
        img.save(output_path)
        
        return {'status': 'success', 'output_path': output_path}
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.task
def generate_report(user_id, report_type, date_range):
    """Generate user report"""
    try:
        # Generate report logic
        report_data = create_report(user_id, report_type, date_range)
        
        # Save report to file
        report_path = f"/reports/{user_id}_{report_type}_{date_range}.pdf"
        save_report_pdf(report_data, report_path)
        
        # Send notification email
        send_email.delay(
            to_email=get_user_email(user_id),
            subject="Your report is ready",
            template_id="report_ready",
            template_data={'download_link': report_path}
        )
        
        return {'status': 'success', 'report_path': report_path}
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Usage in Flask app
@app.route('/send-email', methods=['POST'])
def trigger_email():
    data = request.get_json()
    
    # Queue email task
    task = send_email.delay(
        to_email=data['email'],
        subject=data['subject'],
        template_id=data['template_id'],
        template_data=data.get('template_data', {})
    )
    
    return jsonify({'task_id': task.id})

@app.route('/task-status/<task_id>')
def get_task_status(task_id):
    task = send_email.AsyncResult(task_id)
    
    return jsonify({
        'task_id': task_id,
        'status': task.status,
        'result': task.result if task.ready() else None
    })
```

**Starting Celery Workers:**
```bash
# Start worker for all queues
celery -A myapp worker --loglevel=info

# Start worker for specific queues
celery -A myapp worker --queues=email,media --loglevel=info

# Start worker with concurrency settings
celery -A myapp worker --concurrency=4 --loglevel=info

# Monitor tasks
celery -A myapp flower  # Web-based monitoring
celery -A myapp events  # Command-line monitoring
```