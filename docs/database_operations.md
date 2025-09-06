# Database Operations

## Connection Configuration

Configure your database connection in `config/database.yml`:

```yaml
production:
  adapter: postgresql
  host: db.example.com
  port: 5432
  database: myapp_production
  username: myapp_user
  password: <%= ENV['DATABASE_PASSWORD'] %>
  pool: 20
  timeout: 5000
```

## Migrations

### Running Migrations

```bash
# Run pending migrations
./manage.py migrate

# Run specific migration
./manage.py migrate app_name 0001

# Rollback migration
./manage.py migrate app_name 0000
```

### Creating Migrations

```bash
# Auto-generate migration from model changes
./manage.py makemigrations

# Create empty migration
./manage.py makemigrations --empty app_name
```

## Backup and Restore

### Automated Backups

Backups run daily at 2 AM UTC and are retained for 30 days.

**Manual backup:**
```bash
# Create backup
pg_dump -h localhost -U username -d database_name > backup_$(date +%Y%m%d).sql

# Compressed backup
pg_dump -h localhost -U username -d database_name | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore from Backup

```bash
# Restore from SQL file
psql -h localhost -U username -d database_name < backup_20230115.sql

# Restore from compressed backup
gunzip -c backup_20230115.sql.gz | psql -h localhost -U username -d database_name
```

## Performance Monitoring

### Query Performance

Check slow queries (queries taking >1 second):

```sql
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC;
```

### Connection Monitoring

```sql
-- Current connections
SELECT count(*) as connection_count FROM pg_stat_activity;

-- Connections by state
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;
```

### Index Usage

```sql
-- Tables with missing indexes (high seq_scan/seq_tup_read ratio)
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public' 
ORDER BY n_distinct DESC;
```

## Maintenance Tasks

### Vacuum and Analyze

```bash
# Manual vacuum (run during low traffic)
psql -c "VACUUM ANALYZE;"

# Vacuum specific table
psql -c "VACUUM ANALYZE table_name;"
```

### Reindexing

```bash
# Reindex all indexes
psql -c "REINDEX DATABASE database_name;"

# Reindex specific table
psql -c "REINDEX TABLE table_name;"
```

## Connection Pool Configuration

### PgBouncer Setup

```ini
[databases]
myapp = host=localhost port=5432 dbname=myapp_production

[pgbouncer]
listen_port = 6432
listen_addr = 127.0.0.1
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
server_reset_query = DISCARD ALL
max_client_conn = 100
default_pool_size = 25
```

### Application Pool Settings

For high-traffic applications:
- **Pool size**: 20-30 connections per server
- **Max overflow**: 10 additional connections
- **Pool timeout**: 30 seconds
- **Pool recycle**: 3600 seconds (1 hour)

## Troubleshooting

### Connection Issues

```bash
# Test database connectivity
pg_isready -h localhost -p 5432

# Check if database accepts connections
psql -h localhost -U username -d database_name -c "SELECT 1;"
```

### Lock Monitoring

```sql
-- Check for blocked queries
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity 
  ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
  ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity 
  ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

### Disk Space Monitoring

```bash
# Check database size
psql -c "SELECT pg_size_pretty(pg_database_size('database_name'));"

# Check table sizes
psql -c "SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size FROM pg_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```