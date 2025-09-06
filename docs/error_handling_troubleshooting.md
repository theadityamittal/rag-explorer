# Error Handling & Troubleshooting

## Common Error Messages

### Database Connection Errors

**Error:** `psycopg2.OperationalError: could not connect to server`

**Causes:**
- Database server is down
- Incorrect connection parameters
- Network connectivity issues
- Database server is overloaded

**Solutions:**
```bash
# Check if database is running
pg_isready -h localhost -p 5432

# Test connection with psql
psql -h hostname -U username -d database_name

# Check connection pool settings
# Increase pool size if getting "connection pool exhausted" errors
```

**Error:** `FATAL: remaining connection slots are reserved`

**Solutions:**
```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Kill idle connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' AND state_change < now() - INTERVAL '5 minutes';

-- Increase max_connections in postgresql.conf
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
```

### Authentication Errors

**Error:** `401 Unauthorized - Invalid API key`

**Solutions:**
1. Check API key format: `sk_live_...` or `sk_test_...`
2. Verify key in Authorization header: `Bearer YOUR_KEY`
3. Ensure key hasn't been revoked or expired
4. Check environment variables: `echo $API_KEY`

**Error:** `403 Forbidden - Insufficient permissions`

**Solutions:**
```python
# Check user permissions
def check_permission(user, resource, action):
    permissions = get_user_permissions(user.id)
    required_permission = f"{resource}:{action}"
    return required_permission in permissions

# Common permission patterns:
# user:read, user:write, user:delete
# admin:*, moderator:read, moderator:write
```

### Memory and Performance Errors

**Error:** `MemoryError` or `OutOfMemoryError`

**Causes:**
- Large dataset processing
- Memory leaks in application
- Insufficient server resources

**Solutions:**
```python
# Process data in chunks
def process_large_dataset(data):
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield process_chunk(chunk)

# Use generators instead of lists
def get_users():
    for user in User.objects.all():
        yield user

# Monitor memory usage
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # Force garbage collection
    gc.collect()
```

## Debugging Techniques

### Logging Best Practices

```python
import logging
import traceback
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

def process_user_data(user_id):
    try:
        logger.info(f"Processing user data for user_id: {user_id}")
        
        # Your processing logic here
        user = get_user(user_id)
        result = complex_processing(user)
        
        logger.info(f"Successfully processed user {user_id}, result: {result}")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error for user {user_id}: {e}")
        raise
    except DatabaseError as e:
        logger.error(f"Database error processing user {user_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing user {user_id}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Send to error tracking (Sentry, Rollbar, etc.)
        capture_exception(e, extra={'user_id': user_id})
        raise
```

### Performance Profiling

```python
import cProfile
import pstats
from functools import wraps
import time

def profile(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        pr.disable()
        
        # Print top 10 slowest functions
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        print(f"Total execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@profile
def slow_function():
    # Your slow code here
    pass
```

### Database Query Debugging

```python
import sqlalchemy
from sqlalchemy import event

# Log all SQL queries
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Custom query logging
@event.listens_for(sqlalchemy.engine.Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()
    logger.debug(f"SQL Query: {statement}")
    logger.debug(f"Parameters: {parameters}")

@event.listens_for(sqlalchemy.engine.Engine, "after_cursor_execute")  
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    if total > 0.1:  # Log slow queries (>100ms)
        logger.warning(f"Slow query ({total:.4f}s): {statement[:100]}...")
```

## Error Monitoring & Alerting

### Sentry Integration

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[
        FlaskIntegration(transaction_style='endpoint'),
        SqlalchemyIntegration(),
    ],
    traces_sample_rate=0.1,  # 10% of transactions
    environment="production"
)

# Custom error context
def process_payment(user_id, amount):
    with sentry_sdk.configure_scope() as scope:
        scope.user = {"id": user_id}
        scope.set_tag("payment_amount", amount)
        scope.set_context("payment", {
            "amount": amount,
            "currency": "USD",
            "user_id": user_id
        })
        
        # Your payment logic here
        if amount < 0:
            raise ValueError("Invalid payment amount")
```

### Health Check Endpoints

```python
import time
import psutil
from flask import jsonify

@app.route('/health/detailed')
def detailed_health():
    """Comprehensive health check"""
    checks = {}
    
    # Database connectivity
    try:
        db.session.execute('SELECT 1')
        checks['database'] = {'status': 'healthy', 'response_time_ms': 0}
    except Exception as e:
        checks['database'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Redis connectivity
    try:
        start = time.time()
        redis_client.ping()
        response_time = (time.time() - start) * 1000
        checks['redis'] = {'status': 'healthy', 'response_time_ms': round(response_time, 2)}
    except Exception as e:
        checks['redis'] = {'status': 'unhealthy', 'error': str(e)}
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    checks['system'] = {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': (disk.used / disk.total) * 100,
        'status': 'healthy' if cpu_percent < 80 and memory.percent < 80 else 'degraded'
    }
    
    # Overall health
    all_healthy = all(
        check.get('status') == 'healthy' 
        for check in checks.values() 
        if isinstance(check, dict) and 'status' in check
    )
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'unhealthy',
        'timestamp': time.time(),
        'checks': checks
    }), 200 if all_healthy else 503
```

## Troubleshooting Workflows

### Application Won't Start

**Checklist:**
1. Check environment variables: `printenv | grep -i app`
2. Verify dependencies: `pip freeze | grep -f requirements.txt`
3. Test database connection: `psql $DATABASE_URL -c "SELECT 1;"`
4. Check file permissions: `ls -la config/`
5. Review application logs: `tail -f logs/app.log`
6. Verify port availability: `netstat -tulpn | grep :8000`

### Slow Performance

**Investigation steps:**
```bash
# Check system resources
top
htop
iostat -x 1

# Database performance
psql -c "SELECT query, calls, total_time, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Application profiling
python -m cProfile -o profile.stats your_app.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Network latency
ping api.example.com
traceroute api.example.com
curl -w "@curl-format.txt" -o /dev/null -s "http://api.example.com/endpoint"
```

**curl-format.txt:**
```
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
```

### High Error Rates

**Analysis workflow:**
1. **Check error logs for patterns:**
```bash
# Count error types
grep ERROR app.log | awk '{print $4}' | sort | uniq -c | sort -nr

# Find most frequent errors
grep "Exception:" app.log | sort | uniq -c | sort -nr | head -10
```

2. **Monitor error rates:**
```python
# Error rate calculation
def calculate_error_rate(time_window_minutes=5):
    now = datetime.utcnow()
    start_time = now - timedelta(minutes=time_window_minutes)
    
    total_requests = get_request_count(start_time, now)
    error_requests = get_error_count(start_time, now)
    
    if total_requests == 0:
        return 0
    
    error_rate = (error_requests / total_requests) * 100
    return error_rate

# Alert if error rate > 5%
error_rate = calculate_error_rate()
if error_rate > 5:
    send_alert(f"High error rate: {error_rate:.2f}%")
```

### Memory Leaks

**Detection and debugging:**
```python
import tracemalloc
import gc
import objgraph

# Start memory tracing
tracemalloc.start()

def check_memory_usage():
    """Check for memory leaks"""
    gc.collect()  # Force garbage collection
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    # Show most common object types
    objgraph.show_most_common_types()
    
    # Find objects that weren't garbage collected
    objgraph.show_growth()

# Call periodically to monitor
check_memory_usage()
```

### SSL/TLS Certificate Issues

**Troubleshooting commands:**
```bash
# Check certificate expiration
openssl x509 -in certificate.crt -text -noout | grep "Not After"

# Verify certificate chain
openssl s_client -connect example.com:443 -showcerts

# Check certificate details
curl -vI https://example.com

# Test SSL configuration
openssl s_client -connect example.com:443 -tls1_2
```

**Certificate renewal (Let's Encrypt):**
```bash
# Dry run renewal
certbot renew --dry-run

# Force renewal
certbot renew --force-renewal

# Check certificate status
certbot certificates
```