# GT14 WhaleTracker v14.3 - Deployment Guide

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] Python 3.8 or higher installed
- [ ] MySQL 8.0+ running and accessible
- [ ] Minimum 8GB RAM (16GB recommended)
- [ ] 10GB free disk space for logs and results
- [ ] Network access to data sources

### Database Setup
- [ ] MySQL server running
- [ ] Database `gt14_whaletracker` created
- [ ] Required tables exist:
  - [ ] whale_hourly_complete
  - [ ] whale_alerts_original
  - [ ] universal_features
  - [ ] whale_features_basic
  - [ ] cluster_labels
  - [ ] arima_models
- [ ] User permissions configured

## 🚀 Deployment Steps

### 1. Environment Setup

```bash
# Clone or copy the v14.3 directory
cp -r GT14_v14_3/ /path/to/deployment/

# Navigate to deployment directory
cd /path/to/deployment/GT14_v14_3/

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installations
python -c "import pandas, numpy, mysql.connector, statsmodels; print('Core packages OK')"
```

### 3. Configuration

#### Database Configuration
Edit the database configuration in `GT14_v14_3_FINAL.py`:

```python
self.db_config = {
    'host': 'your_mysql_host',
    'user': 'your_mysql_user',
    'password': 'your_mysql_password',
    'database': 'gt14_whaletracker',
    'pool_size': 5,
    'pool_reset_session': True
}
```

#### Logging Configuration
Update log paths if needed:

```python
self.log_dir = Path("/path/to/logs/GT14_v14_3")
self.general_logs = Path("/path/to/general/logs")
```

### 4. Initial Data Setup

```bash
# Test database connection
python -c "from GT14_v14_3_FINAL import GT14_Complete_Pipeline; p = GT14_Complete_Pipeline(); print('DB connection OK')"

# Run feature persistence to populate features
python feature_persistence_quick.py

# Verify features loaded
mysql -u root -p gt14_whaletracker -e "SELECT COUNT(*) FROM whale_features_basic;"
```

### 5. Test Deployment

```bash
# Run comprehensive tests
python tests/test_complete_pipeline.py

# Run a single analysis stage test
python -c "from GT14_v14_3_FINAL import GT14_Complete_Pipeline; p = GT14_Complete_Pipeline(); p.load_and_analyze_data()"
```

### 6. Production Deployment

#### Option A: Direct Execution
```bash
# Full pipeline execution
python GT14_v14_3_FINAL.py

# With logging to file
python GT14_v14_3_FINAL.py > execution_log.txt 2>&1
```

#### Option B: Scheduled Execution (Cron)
```bash
# Add to crontab for hourly execution
0 * * * * cd /path/to/GT14_v14_3 && /path/to/venv/bin/python GT14_v14_3_FINAL.py
```

#### Option C: Service/Daemon
Create a systemd service file `/etc/systemd/system/gt14-whaletracker.service`:

```ini
[Unit]
Description=GT14 WhaleTracker v14.3
After=mysql.service

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/GT14_v14_3
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python GT14_v14_3_FINAL.py
Restart=on-failure
RestartSec=300

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable gt14-whaletracker
sudo systemctl start gt14-whaletracker
```

## 🔍 Monitoring

### Log Files
Monitor these log locations:
- `./logs/analysis_log_*.log` - Main execution logs
- `./logs/analysis_metrics_*.json` - Performance metrics
- `./logs/terminal_output_*.log` - Console output

### Health Checks
```python
# check_health.py
from GT14_v14_3_FINAL import GT14_Complete_Pipeline
import mysql.connector

def check_system_health():
    checks = {
        'database': False,
        'features': False,
        'models': False
    }
    
    try:
        # Check DB connection
        pipeline = GT14_Complete_Pipeline()
        conn = mysql.connector.connect(**pipeline.db_config)
        checks['database'] = conn.is_connected()
        
        # Check features
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM whale_features_basic")
        count = cursor.fetchone()[0]
        checks['features'] = count > 0
        
        # Check models
        cursor.execute("SELECT COUNT(*) FROM arima_models")
        count = cursor.fetchone()[0]
        checks['models'] = count > 0
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Health check error: {e}")
    
    return checks

if __name__ == "__main__":
    health = check_system_health()
    print(f"System Health: {health}")
```

### Performance Monitoring
```bash
# Monitor resource usage
htop
mysqladmin -u root -p processlist
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 2. MySQL Connection Pool Exhausted
```python
# Increase pool size in configuration
'pool_size': 10,  # Increase from 5
'max_overflow': 5  # Allow overflow connections
```

#### 3. Slow Performance
- Check MySQL indexes:
```sql
CREATE INDEX idx_timestamp ON whale_hourly_complete(timestamp);
CREATE INDEX idx_timestamp_price ON whale_hourly_complete(timestamp, btc_price);
```

- Enable query caching in MySQL configuration

#### 4. Disk Space Issues
```bash
# Clean old logs and results
find ./logs -name "*.log" -mtime +7 -delete
find ./results_* -mtime +30 -delete
```

## 🔄 Updates and Maintenance

### Regular Maintenance Tasks
1. **Daily**: Check logs for errors
2. **Weekly**: Clean old result files
3. **Monthly**: Optimize MySQL tables
4. **Quarterly**: Review and update feature selections

### Update Procedure
1. Backup current deployment
2. Test updates in staging environment
3. Schedule maintenance window
4. Deploy updates
5. Run health checks
6. Monitor for 24 hours

## 📊 Performance Tuning

### MySQL Optimization
```sql
-- Add to my.cnf
[mysqld]
innodb_buffer_pool_size = 4G
query_cache_size = 256M
query_cache_type = 1
max_connections = 200
```

### Python Optimization
```python
# Use multiprocessing for parallel stages
from multiprocessing import Pool

def run_parallel_analyses():
    with Pool(processes=4) as pool:
        results = pool.map(analyze_stage, stages)
```

### Resource Limits
```bash
# Set ulimits for production
ulimit -n 4096  # Increase file descriptors
ulimit -m unlimited  # Remove memory limits
```

## 🔐 Security Considerations

1. **Database Credentials**: Use environment variables
```python
import os
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'password': os.getenv('DB_PASSWORD'),
}
```

2. **File Permissions**:
```bash
chmod 600 config_files
chmod 755 scripts
chmod 644 logs/*
```

3. **Network Security**:
- Use SSL for MySQL connections
- Restrict database access by IP
- Use firewall rules for service ports

## 📝 Post-Deployment Verification

Run this verification script after deployment:

```python
# verify_deployment.py
import sys
from pathlib import Path

def verify_deployment():
    checks = []
    
    # Check Python version
    checks.append(('Python 3.8+', sys.version_info >= (3, 8)))
    
    # Check required modules
    required_modules = [
        'pandas', 'numpy', 'mysql.connector',
        'statsmodels', 'sklearn', 'matplotlib'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            checks.append((f'{module} installed', True))
        except ImportError:
            checks.append((f'{module} installed', False))
    
    # Check directories
    for dir_name in ['logs', 'tests']:
        checks.append((f'{dir_name} directory', Path(dir_name).exists()))
    
    # Check main files
    for file_name in ['GT14_v14_3_FINAL.py', 'requirements.txt']:
        checks.append((f'{file_name} exists', Path(file_name).exists()))
    
    # Print results
    print("Deployment Verification Results:")
    print("-" * 40)
    for check, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:<30} {status}")
    
    return all(passed for _, passed in checks)

if __name__ == "__main__":
    success = verify_deployment()
    sys.exit(0 if success else 1)
```

---

## 📞 Support

For deployment issues:
1. Check logs in `./logs/` directory
2. Run health check script
3. Verify database connectivity
4. Review error messages in terminal output

Remember to backup your data before any major deployment changes!