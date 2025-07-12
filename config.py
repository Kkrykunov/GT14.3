"""
Конфігураційний файл для GT14 v14.2
Скопіюйте цей файл як config.py та налаштуйте під ваші параметри
"""

# MySQL конфігурація
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'whale_tracker_2024',
    'database': 'gt14_whaletracker'
}

# Параметри кластеризації
CLUSTERING_CONFIG = {
    'n_clusters': 7,  # Кількість кластерів
    'method': 'KMeans',  # KMeans, DBSCAN, AgglomerativeClustering
    'auto_optimize': True,  # Автоматична оптимізація кількості кластерів
    'min_clusters': 2,
    'max_clusters': 15
}

# Параметри ARIMA
ARIMA_CONFIG = {
    'target_mape': 12.0,  # Цільова точність в %
    'forecast_horizon': 24,  # Горизонт прогнозу в годинах
    'validation_split': 0.8,  # Розділення на навчальну/тестову вибірки
    'seasonal_periods': [24, 168]  # Сезонні періоди (доба, тиждень)
}

# Параметри універсальних фічей
FEATURE_CONFIG = {
    'lag_periods': [1, 3, 6, 12, 24, 48],  # Лагові періоди
    'rolling_windows': [3, 6, 12, 24, 48, 168],  # Вікна для rolling statistics
    'technical_indicators': ['RSI', 'MACD', 'BB', 'STOCH', 'WILLIAMS'],
    'batch_size': 1000  # Розмір батчу для збереження в MySQL
}

# Шляхи до файлів
PATHS = {
    'results_dir': 'results',
    'logs_dir': 'logs',
    'models_dir': 'models',
    'visualizations_dir': 'visualizations'
}

# Параметри візуалізації
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'save_format': 'png'
}

# Параметри логування
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_to_file': True
}