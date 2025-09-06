# Performance Tuning

## Database Optimization

### Query Optimization

**Identify Slow Queries:**
```sql
-- Enable query logging in PostgreSQL
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1 second

-- Find slowest queries
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 20;

-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM users WHERE email = 'user@example.com';
```

**Index Optimization:**
```sql
-- Find missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats 
WHERE schemaname = 'public' 
AND n_distinct > 100
AND correlation < 0.1;

-- Create composite indexes for common query patterns
CREATE INDEX idx_users_email_active ON users(email, is_active) 
WHERE is_active = true;

-- Partial indexes for filtered queries
CREATE INDEX idx_orders_pending ON orders(created_at) 
WHERE status = 'pending';

-- Check index usage
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
ORDER BY idx_tup_read DESC;
```

**Connection Pool Tuning:**
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Optimized connection pool settings
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,              # Core connections
    max_overflow=30,           # Additional connections
    pool_pre_ping=True,        # Validate connections
    pool_recycle=3600,         # Recycle connections every hour
    pool_timeout=30,           # Wait time for connection
    echo=False                 # Disable SQL logging in production
)

# Monitor connection pool
def get_pool_status():
    pool = engine.pool
    return {
        'size': pool.size(),
        'checked_in': pool.checkedin(),
        'checked_out': pool.checkedout(),
        'overflow': pool.overflow(),
        'invalid': pool.invalid()
    }
```

### Database Configuration

**PostgreSQL Settings (postgresql.conf):**
```ini
# Memory settings (adjust for your server)
shared_buffers = 4GB              # 25% of RAM
effective_cache_size = 12GB       # 75% of RAM
work_mem = 64MB                   # Per query operation
maintenance_work_mem = 1GB        # For VACUUM, CREATE INDEX

# Connection settings
max_connections = 200
superuser_reserved_connections = 3

# WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 200ms

# Query planner
random_page_cost = 1.1           # For SSD storage
effective_io_concurrency = 200   # For SSD storage

# Logging
log_min_duration_statement = 1000  # Log slow queries
log_lock_waits = on
log_temp_files = 10MB
```

## Caching Strategies

### Redis Caching

```python
import redis
import json
import pickle
from functools import wraps
from datetime import timedelta

class CacheManager:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url, decode_responses=True)
    
    def cache_result(self, key_prefix, ttl=3600):
        """Decorator to cache function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = self.redis.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.redis.setex(cache_key, ttl, json.dumps(result))
                return result
            return wrapper
        return decorator
    
    def invalidate_pattern(self, pattern):
        """Invalidate cache keys matching pattern"""
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

# Usage examples
cache = CacheManager('redis://localhost:6379')

@cache.cache_result('user_profile', ttl=1800)  # 30 minutes
def get_user_profile(user_id):
    return expensive_database_query(user_id)

@cache.cache_result('search_results', ttl=300)  # 5 minutes
def search_products(query, filters):
    return complex_search_query(query, filters)

# Manual caching
def get_popular_products():
    cache_key = 'popular_products'
    products = cache.redis.get(cache_key)
    
    if not products:
        products = Product.objects.filter(is_popular=True).all()
        cache.redis.setex(cache_key, 3600, json.dumps(products))
    else:
        products = json.loads(products)
    
    return products
```

### HTTP Caching

```python
from flask import Flask, request, make_response
import hashlib
from datetime import datetime, timedelta

app = Flask(__name__)

def add_cache_headers(response, max_age=3600, etag=None):
    """Add HTTP cache headers to response"""
    response.headers['Cache-Control'] = f'public, max-age={max_age}'
    response.headers['Expires'] = (datetime.utcnow() + timedelta(seconds=max_age)).strftime('%a, %d %b %Y %H:%M:%S GMT')
    
    if etag:
        response.headers['ETag'] = etag
    
    return response

@app.route('/api/products')
def get_products():
    # Check if-none-match header for ETag
    client_etag = request.headers.get('If-None-Match')
    
    # Generate content
    products = get_products_from_db()
    content = json.dumps(products)
    
    # Generate ETag based on content
    etag = hashlib.md5(content.encode()).hexdigest()
    
    # Return 304 if content hasn't changed
    if client_etag == etag:
        return '', 304
    
    response = make_response(content)
    response.headers['Content-Type'] = 'application/json'
    return add_cache_headers(response, max_age=300, etag=etag)

# Cache invalidation
def invalidate_product_cache():
    """Invalidate product-related caches"""
    cache.invalidate_pattern('popular_products*')
    cache.invalidate_pattern('search_results*')
```

## Application Performance

### Code Optimization

```python
import time
import cProfile
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# Performance timing decorator
def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Memoization for expensive computations
@lru_cache(maxsize=128)
def expensive_calculation(n):
    # Some expensive computation
    return sum(i ** 2 for i in range(n))

# Batch processing for database operations
def batch_process_users(user_ids, batch_size=100):
    """Process users in batches to avoid memory issues"""
    results = []
    
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i:i + batch_size]
        batch_results = User.objects.filter(id__in=batch).all()
        results.extend(process_user_batch(batch_results))
        
        # Yield control to prevent blocking
        time.sleep(0.01)
    
    return results

# Parallel processing for I/O bound tasks
def process_urls_parallel(urls, max_workers=10):
    """Process multiple URLs concurrently"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    return results

# Memory-efficient data processing
def process_large_dataset(data_source):
    """Generator-based processing for large datasets"""
    for batch in data_source.iter_batches(size=1000):
        processed_batch = []
        
        for item in batch:
            processed_item = expensive_transform(item)
            processed_batch.append(processed_item)
            
            # Yield periodically to avoid memory buildup
            if len(processed_batch) >= 100:
                yield processed_batch
                processed_batch = []
        
        # Yield remaining items
        if processed_batch:
            yield processed_batch
```

### Async/Await Optimization

```python
import asyncio
import aiohttp
import asyncpg
from typing import List

class AsyncDatabasePool:
    def __init__(self, database_url, min_size=10, max_size=20):
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.pool = None
    
    async def init_pool(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.min_size,
            max_size=self.max_size,
            command_timeout=60
        )
    
    async def fetch_users(self, user_ids: List[int]):
        """Fetch multiple users concurrently"""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM users WHERE id = ANY($1)"
            return await conn.fetch(query, user_ids)
    
    async def close_pool(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()

async def fetch_external_data(urls: List[str]) -> List[dict]:
    """Fetch data from multiple URLs concurrently"""
    results = []
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=aiohttp.TCPConnector(limit=100)
    ) as session:
        
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch_single_url(session, url))
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, Exception):
                print(f"Error: {response}")
            else:
                results.append(response)
    
    return results

async def fetch_single_url(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch single URL with retry logic"""
    for attempt in range(3):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            await asyncio.sleep(1 ** attempt)  # Exponential backoff
```

## Memory Management

### Memory Profiling

```python
import psutil
import gc
import sys
import tracemalloc
from memory_profiler import profile

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent(),
        'available': psutil.virtual_memory().available / 1024 / 1024  # MB
    }

@profile  # Use with: pip install memory_profiler
def memory_intensive_function():
    """Function to profile memory usage"""
    data = []
    for i in range(100000):
        data.append({'id': i, 'value': f'item_{i}'})
    
    # Process data
    result = [item for item in data if item['id'] % 2 == 0]
    
    # Clean up
    del data
    gc.collect()
    
    return result

# Memory leak detection
def check_for_memory_leaks():
    """Check for potential memory leaks"""
    gc.collect()  # Force garbage collection
    
    # Get object counts by type
    object_counts = {}
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
    
    # Sort by count
    sorted_counts = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 object types by count:")
    for obj_type, count in sorted_counts[:10]:
        print(f"{obj_type}: {count}")

# Memory-efficient data structures
class CircularBuffer:
    """Memory-efficient circular buffer"""
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.is_full = False
    
    def put(self, item):
        self.buffer[self.head] = item
        
        if self.is_full:
            self.tail = (self.tail + 1) % self.size
        
        self.head = (self.head + 1) % self.size
        
        if self.head == self.tail:
            self.is_full = True
    
    def get(self):
        if self.head == self.tail and not self.is_full:
            return None
        
        item = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.size
        self.is_full = False
        
        return item
```

## Load Testing & Monitoring

### Load Testing with Locust

```python
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Login user at start"""
        response = self.client.post("/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        
        if response.status_code == 200:
            self.token = response.json()["token"]
        else:
            self.token = None
    
    @task(3)  # Weight: 3 (more frequent)
    def view_products(self):
        """Test product listing endpoint"""
        self.client.get("/api/products")
    
    @task(2)  # Weight: 2
    def search_products(self):
        """Test search endpoint"""
        self.client.get("/api/products/search?q=test")
    
    @task(1)  # Weight: 1 (less frequent)
    def create_order(self):
        """Test order creation"""
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            self.client.post("/api/orders", json={
                "items": [{"product_id": 1, "quantity": 2}]
            }, headers=headers)

# Run with: locust -f load_test.py --host=http://localhost:8000
```

### Performance Monitoring

```python
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MetricPoint:
    timestamp: float
    value: float

class PerformanceMonitor:
    def __init__(self, window_size=300):  # 5 minutes
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def record_metric(self, name: str, value: float):
        """Record a metric value"""
        now = time.time()
        point = MetricPoint(now, value)
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque()
            
            self.metrics[name].append(point)
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [point.value for point in self.metrics[name]]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1]
            }
    
    def _cleanup_old_metrics(self):
        """Remove old metric points"""
        while True:
            time.sleep(60)  # Check every minute
            cutoff_time = time.time() - self.window_size
            
            with self.lock:
                for name, points in self.metrics.items():
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()

# Usage
monitor = PerformanceMonitor()

# Middleware to track request performance
def performance_middleware(app):
    def middleware(environ, start_response):
        start_time = time.time()
        
        def custom_start_response(status, headers, exc_info=None):
            # Record response time
            response_time = (time.time() - start_time) * 1000  # ms
            monitor.record_metric('response_time', response_time)
            
            # Record status code
            status_code = int(status.split()[0])
            monitor.record_metric(f'status_{status_code}', 1)
            
            return start_response(status, headers, exc_info)
        
        return app(environ, custom_start_response)
    
    return middleware

# API endpoint to get metrics
@app.route('/metrics/performance')
def get_performance_metrics():
    return jsonify({
        'response_time': monitor.get_metric_stats('response_time'),
        'status_200': monitor.get_metric_stats('status_200'),
        'status_404': monitor.get_metric_stats('status_404'),
        'status_500': monitor.get_metric_stats('status_500')
    })
```

## CDN and Static Asset Optimization

### Asset Optimization

```python
import gzip
import os
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Gzip compression middleware
def gzip_response(response):
    """Compress response if client accepts gzip"""
    accept_encoding = request.headers.get('Accept-Encoding', '')
    
    if 'gzip' not in accept_encoding.lower():
        return response
    
    if response.status_code < 200 or response.status_code >= 300:
        return response
    
    # Compress response data
    compressed_data = gzip.compress(response.get_data())
    
    response.set_data(compressed_data)
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(compressed_data)
    
    return response

@app.after_request
def after_request(response):
    return gzip_response(response)

# Static file serving with caching
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files with proper cache headers"""
    filename = secure_filename(filename)
    file_path = os.path.join(app.static_folder, filename)
    
    if not os.path.exists(file_path):
        return "File not found", 404
    
    # Set cache headers based on file type
    if filename.endswith(('.css', '.js')):
        cache_timeout = 86400 * 30  # 30 days
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico')):
        cache_timeout = 86400 * 7   # 7 days
    else:
        cache_timeout = 86400       # 1 day
    
    response = send_file(file_path, cache_timeout=cache_timeout)
    
    # Add additional headers
    response.headers['Cache-Control'] = f'public, max-age={cache_timeout}'
    
    return response
```