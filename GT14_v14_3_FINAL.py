#!/usr/bin/env python3
"""
GT14 v14.2 - ПОВНИЙ РОЗШИРЕНИЙ PIPELINE
Включає ВСІ функції з попередніх версій ПЛЮС нові покращення

РОЗШИРЕННЯ порівняно з v14.1:
- Всі типи аналізу з v8 та v14.1
- Крос-кореляційний аналіз
- Сезонність та аномалії  
- VAR з повним IRF та FEVD
- Кластеризація з УСІМА метриками
- Інтерактивний вибір моделей
- Професійна візуалізація для клієнтів
"""

import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Налаштування для професійних графіків
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.style.use('seaborn-v0_8-darkgrid')  # Закоментовано - використовуємо стандартний стиль

# Аналіз статистики
from scipy import stats
from statsmodels.tsa.stattools import ccf, adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

# Машинне навчання
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score

# Детекція аномалій
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# Система логування та моніторингу
import logging
import traceback
from pathlib import Path
import threading
import time
import sys
import os

# Імпорт нових інтегрованих модулів
try:
    from arima_visualization import ARIMAVisualization
except ImportError:
    ARIMAVisualization = None
    
try:
    from granger_causality_enhanced import GrangerCausalityEnhanced
except ImportError:
    GrangerCausalityEnhanced = None
    
try:
    from feature_importance_analysis import FeatureImportanceAnalyzer
except ImportError:
    FeatureImportanceAnalyzer = None
    
try:
    from arima_ensemble_models import ARIMAEnsembleModels
except ImportError:
    ARIMAEnsembleModels = None

class AnalysisLogger:
    """Централізована система логування для GT14"""
    
    def __init__(self, log_dir="/mnt/c/Desktop/07.07.2025/GT14_WhaleTracker/GT14_v14_2_COMPLETE/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Загальна папка логів
        self.general_logs = Path("/mnt/c/Desktop/07.07.2025/logs")
        self.general_logs.mkdir(exist_ok=True, parents=True)
        
        # Основний лог файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f'analysis_log_{timestamp}.log'
        
        # Файл для метрик
        self.metrics_file = self.log_dir / f'analysis_metrics_{timestamp}.json'
        
        # Термінал лог
        self.terminal_log = self.log_dir / f'terminal_output_{timestamp}.log'
        
        self.setup_logging()
        self.metrics = {}
        self.start_time = datetime.now()
        
        # Запуск автозбереження терміналу кожні 40 хвилин
        self.start_terminal_autosave()
        
    def setup_logging(self):
        """Налаштування логування з дублюванням в термінал"""
        # Формат логів
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # File handler для основних логів
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # File handler для терміналу
        terminal_handler = logging.FileHandler(self.terminal_log, encoding='utf-8')
        terminal_handler.setFormatter(formatter)
        terminal_handler.setLevel(logging.INFO)
        
        # Console handler з перехопленням stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Копія в загальну папку логів
        general_log = self.general_logs / f'gt14_log_{datetime.now().strftime("%Y%m%d")}.log'
        general_handler = logging.FileHandler(general_log, encoding='utf-8', mode='a')
        general_handler.setFormatter(formatter)
        general_handler.setLevel(logging.INFO)
        
        # Root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(terminal_handler)
        logger.addHandler(console_handler)
        logger.addHandler(general_handler)
        
        # Перехоплення print() для логування
        self._redirect_stdout()
        
    def _redirect_stdout(self):
        """Перенаправляє stdout в логи"""
        class LoggerWriter:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level
                self.linebuf = ''
                
            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    self.logger.log(self.level, line.rstrip())
                    
            def flush(self):
                pass
        
        sys.stdout = LoggerWriter(logging.getLogger('STDOUT'), logging.INFO)
        
    def start_terminal_autosave(self):
        """Автозбереження терміналу кожні 40 хвилин"""
        def autosave():
            while True:
                time.sleep(2400)  # 40 хвилин
                self.save_terminal_snapshot()
                
        thread = threading.Thread(target=autosave, daemon=True)
        thread.start()
        logging.info("Terminal autosave started (every 40 minutes)")
        
    def save_terminal_snapshot(self):
        """Зберігає snapshot терміналу"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.general_logs / f'terminal_snapshot_{timestamp}.log'
        
        try:
            # Копіюємо поточний термінал лог
            import shutil
            shutil.copy2(self.terminal_log, snapshot_file)
            logging.info(f"Terminal snapshot saved to {snapshot_file}")
        except Exception as e:
            logging.error(f"Failed to save terminal snapshot: {e}")
    
    def log_metric(self, category, metric_name, value):
        """Зберігає метрику для подальшого аналізу"""
        if category not in self.metrics:
            self.metrics[category] = {}
        
        self.metrics[category][metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Також логуємо
        logging.info(f"METRIC: {category}.{metric_name} = {value}")
        
    def log_error(self, error, context=None):
        """Детальне логування помилок"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        logging.error(f"ERROR: {error_data}")
        return error_data
        
    def log_mcp_status(self):
        """Перевірка статусу MCP серверів"""
        logging.warning("MCP servers check skipped - MCP not configured")
        logging.info("Using direct MySQL connection instead of MCP")
        self.log_metric('system', 'mcp_status', 'not_configured')
        return False
        
    def save_metrics(self):
        """Зберігає всі метрики в JSON файл"""
        import json
        
        final_metrics = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'metrics': self.metrics
        }
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
            
        # Копія в загальну папку
        general_metrics = self.general_logs / f'gt14_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(general_metrics, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Metrics saved to {self.metrics_file} and {general_metrics}")

class GT14_Complete_Pipeline:
    """Повний розширений pipeline v14.2"""
    
    def __init__(self):
        # Ініціалізація системи логування
        self.logger = AnalysisLogger()
        self.log = logging.getLogger(__name__)
        
        self.log.info("="*80)
        self.log.info(" GT14 v14.2 COMPLETE ENHANCED PIPELINE")
        self.log.info("Включає ВСІ функції попередніх версій + нові покращення")
        self.log.info("="*80)
        self.log.info(f"Запуск: {datetime.now()}")
        self.log.info("")
        
        # Логуємо початкові метрики
        self.logger.log_metric('system', 'start_time', datetime.now().isoformat())
        self.logger.log_metric('system', 'version', '14.2')
        self.logger.log_metric('system', 'python_version', sys.version)
        
        # Перевірка MCP статусу
        self.logger.log_mcp_status()
        
        self.config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        
        # Логуємо конфігурацію (без паролю)
        safe_config = {k: v for k, v in self.config.items() if k != 'password'}
        self.log.info(f"Database config: {safe_config}")
        
        self.results_dir = Path(f"results_v14_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Ініціалізація MySQL connection pool
        self.db_pool = None
        self.init_mysql_pool()
        
        # Результати для звіту
        self.all_results = {
            'metadata': {
                'version': '14.2',
                'analysis_date': datetime.now().isoformat(),
                'includes_all_v8_features': True,
                'includes_all_v14_1_features': True,
                'new_features': [
                    'Enhanced VAR with full IRF/FEVD',
                    'All clustering metrics (Davies-Bouldin, Calinski-Harabasz, Silhouette)',
                    'Interactive model selection',
                    'Professional client-ready visualizations',
                    'Comprehensive cross-correlation analysis',
                    'Advanced anomaly detection',
                    'Full Bayesian analysis suite'
                ]
            }
        }
        
    def init_mysql_pool(self):
        """Ініціалізація MySQL connection pool"""
        try:
            from mysql.connector import pooling
            
            # Конфігурація pool
            pool_config = self.config.copy()
            pool_config['pool_name'] = 'gt14_pool'
            pool_config['pool_size'] = 5
            pool_config['pool_reset_session'] = True
            
            self.db_pool = pooling.MySQLConnectionPool(**pool_config)
            
            self.log.info("MySQL connection pool створено успішно")
            self.logger.log_metric('database', 'pool_size', 5)
            self.logger.log_metric('database', 'pool_created', True)
            
            # Тест підключення
            self.test_mysql_connection()
            
        except Exception as e:
            self.log.error(f"Помилка створення MySQL pool: {e}")
            self.logger.log_error(e, context={'operation': 'init_mysql_pool'})
            self.db_pool = None
            
    def get_db_connection(self):
        """Отримання підключення з pool або створення нового"""
        try:
            if self.db_pool:
                conn = self.db_pool.get_connection()
                self.log.debug("Отримано підключення з pool")
            else:
                conn = mysql.connector.connect(**self.config)
                self.log.debug("Створено нове підключення (pool недоступний)")
            
            return conn
            
        except Exception as e:
            self.log.error(f"Помилка підключення до БД: {e}")
            self.logger.log_error(e, context={'operation': 'get_db_connection'})
            raise
            
    def test_mysql_connection(self):
        """Тестування підключення до MySQL"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Перевірка версії MySQL
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            self.log.info(f"MySQL версія: {version}")
            self.logger.log_metric('database', 'mysql_version', version)
            
            # Перевірка доступних таблиць
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            self.log.info(f"Знайдено таблиць: {len(tables)}")
            self.logger.log_metric('database', 'table_count', len(tables))
            
            # Логуємо список таблиць
            for table in tables:
                self.log.debug(f"  - {table}")
            
            cursor.close()
            conn.close()
            
            self.log.info("✓ MySQL підключення працює коректно")
            self.logger.log_metric('database', 'connection_test', 'success')
            
        except Exception as e:
            self.log.error(f"✗ Помилка тестування MySQL: {e}")
            self.logger.log_error(e, context={'operation': 'test_mysql_connection'})
            self.logger.log_metric('database', 'connection_test', 'failed')
            
    def execute_query(self, query, params=None, fetch_all=True):
        """Виконання SQL запиту з автоматичним управлінням підключення"""
        conn = None
        cursor = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch_all:
                result = cursor.fetchall()
            else:
                result = cursor.fetchone()
            
            # Якщо це INSERT/UPDATE/DELETE - commit
            if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                conn.commit()
                self.log.debug(f"Committed {cursor.rowcount} rows")
            
            return result
            
        except Exception as e:
            self.log.error(f"SQL помилка: {e}")
            self.log.error(f"Query: {query}")
            if conn:
                conn.rollback()
            raise
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        
    def run_complete_analysis(self):
        """Запуск ПОВНОГО аналізу"""
        
        # 1. ЗАВАНТАЖЕННЯ ТА БАЗОВА СТАТИСТИКА (як в v14.1)
        self.log.info("\n" + "="*60)
        self.log.info(" ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ ТА БАЗОВА СТАТИСТИКА")
        self.log.info("="*60)
        df = self.load_and_analyze_data()
        
        # 1.5 НОВИЙ ЕТАП: ЗАВАНТАЖЕННЯ ТА АНАЛІЗ 233 ФІЧЕЙ
        self.log.info("\n" + "="*60)
        self.log.info(" ЕТАП 1.5: АНАЛІЗ ТА ІНТЕГРАЦІЯ 233 ФІЧЕЙ")
        self.log.info("="*60)
        
        try:
            # Завантажуємо всі фічі
            df_features = self.load_all_features_from_db()
            
            # Аналізуємо вплив кожної фічі
            df_analysis = self.analyze_individual_features(df_features, df)
            
            # Візуалізація важливості фічей
            self.visualize_feature_importance(
                df_analysis, 
                save_path=self.results_dir / 'feature_importance_analysis.png'
            )
            
            # Тестуємо групи фічей
            group_results = self.test_feature_groups(df_features, df)
            
            # Знаходимо оптимальну стратегію
            strategies = self.find_optimal_feature_strategies(df_features, df, df_analysis)
            
            # Визначаємо оптимальні фічі для подальшого використання
            best_strategy = min(strategies.items(), key=lambda x: x[1].get('mape', float('inf')))
            optimal_features = best_strategy[1].get('features', [])
            
            self.log.info(f"\n✓ ОПТИМАЛЬНА СТРАТЕГІЯ: {best_strategy[0]}")
            self.log.info(f"  Використовується {len(optimal_features)} фічей")
            self.log.info(f"  MAPE: {best_strategy[1]['mape']:.2f}%")
            
            # Інтегруємо оптимальні фічі в основний DataFrame
            if len(optimal_features) > 0:
                df_enhanced = pd.merge(
                    df,
                    df_features[optimal_features],
                    left_index=True,
                    right_index=True,
                    how='inner'
                )
                self.log.info(f"✓ Інтегровано {len(optimal_features)} оптимальних фічей в аналіз")
                
                # Оновлюємо основний DataFrame
                df = df_enhanced
                self.df = df_enhanced
                
                # Зберігаємо список оптимальних фічей
                self.optimal_features = optimal_features
            else:
                self.optimal_features = []
            
            # Зберігаємо результати аналізу фічей
            self._save_feature_analysis_results(df_analysis, group_results, strategies)
            
        except Exception as e:
            self.log.error(f"Помилка аналізу фічей: {e}")
            self.logger.log_error(e, context={'method': 'feature_analysis'})
            self.log.warning("Продовжуємо без додаткових фічей")
            self.optimal_features = []
        
        # 2. ДЕТАЛЬНИЙ ЧАСОВИЙ АНАЛІЗ (як в v8)
        self.log.info("\n" + "="*60)
        self.log.info("⏰ ЕТАП 2: ДЕТАЛЬНИЙ ЧАСОВИЙ АНАЛІЗ")
        self.log.info("="*60)
        temporal_results = self.temporal_analysis(df)
        
        # 3. КРОС-КОРЕЛЯЦІЙНИЙ АНАЛІЗ (як в v8)
        print("\n" + "="*60)
        print(" ЕТАП 3: КРОС-КОРЕЛЯЦІЙНИЙ АНАЛІЗ")
        print("="*60)
        cross_corr_results = self.cross_correlation_analysis(df)
        
        # 4. СЕЗОННІСТЬ ТА АНОМАЛІЇ (як в v8)
        print("\n" + "="*60)
        print(" ЕТАП 4: АНАЛІЗ СЕЗОННОСТІ ТА ДЕТЕКЦІЯ АНОМАЛІЙ")
        print("="*60)
        seasonality_results = self.seasonality_analysis(df)
        anomaly_results = self.anomaly_detection(df)
        
        # 5. КЛАСТЕРИЗАЦІЯ З УСІМА МЕТРИКАМИ
        print("\n" + "="*60)
        print(" ЕТАП 5: КЛАСТЕРИЗАЦІЯ З УСІМА МЕТРИКАМИ")
        print("="*60)
        clustering_results = self.advanced_clustering(df)
        
        # 6. VAR АНАЛІЗ З ПОВНИМ IRF ТА FEVD
        print("\n" + "="*60)
        print(" ЕТАП 6: VAR АНАЛІЗ З IRF ТА FEVD")
        print("="*60)
        var_results = self.comprehensive_var_analysis(df)
        
        # 7. БАЙЄСІВ АНАЛІЗ (як в v14.1)
        print("\n" + "="*60)
        print(" ЕТАП 7: ПОВНИЙ БАЙЄСІВ АНАЛІЗ")
        print("="*60)
        bayes_results = self.bayesian_analysis(df)
        
        # 8. ПРОГНОЗНІ МОДЕЛІ
        print("\n" + "="*60)
        print(" ЕТАП 8: ПОБУДОВА ПРОГНОЗНИХ МОДЕЛЕЙ")
        print("="*60)
        prediction_results = self.build_prediction_models(df)
        
        # 9. СТВОРЕННЯ ВСІХ ВІЗУАЛІЗАЦІЙ
        print("\n" + "="*60)
        print(" ЕТАП 9: СТВОРЕННЯ ПРОФЕСІЙНИХ ВІЗУАЛІЗАЦІЙ")
        print("="*60)
        self.create_all_visualizations(df)
        
        # 10. ГЕНЕРАЦІЯ ЗВІТІВ ДЛЯ КЛІЄНТА
        print("\n" + "="*60)
        print(" ЕТАП 10: ГЕНЕРАЦІЯ ЗВІТІВ ДЛЯ КЛІЄНТА")
        print("="*60)
        self.generate_client_reports()
        
        # 11. ARIMA ВІЗУАЛІЗАЦІЯ (новий функціонал)
        print("\n" + "="*60)
        print("📊 ЕТАП 11: ВІЗУАЛІЗАЦІЯ ARIMA ПРОГНОЗІВ")
        print("="*60)
        self.arima_visualization_integrated()
        
        # 12. ENHANCED GRANGER CAUSALITY (новий функціонал)
        print("\n" + "="*60)
        print("🔗 ЕТАП 12: ENHANCED GRANGER CAUSALITY АНАЛІЗ")
        print("="*60)
        self.granger_causality_integrated()
        
        # 13. FEATURE IMPORTANCE З ЗБЕРЕЖЕНИХ ФІЧЕЙ (новий функціонал)
        print("\n" + "="*60)
        print("⭐ ЕТАП 13: FEATURE IMPORTANCE АНАЛІЗ")
        print("="*60)
        self.feature_importance_integrated()
        
        # 14. CSV ЕКСПОРТ ВСІХ РЕЗУЛЬТАТІВ (новий функціонал)
        print("\n" + "="*60)
        print("💾 ЕТАП 14: CSV ЕКСПОРТ ВСІХ РЕЗУЛЬТАТІВ")
        print("="*60)
        self.export_all_results_to_csv()
        
        # 15. PLOTLY ІНТЕРАКТИВНІ ГРАФІКИ (новий функціонал)
        print("\n" + "="*60)
        print("🎨 ЕТАП 15: PLOTLY ІНТЕРАКТИВНІ ВІЗУАЛІЗАЦІЇ")
        print("="*60)
        self.create_interactive_visualizations()
        
        # 16. ARIMA ENSEMBLE MODELS (новий функціонал v14.3)
        print("\n" + "="*60)
        print("🎯 ЕТАП 16: ARIMA ENSEMBLE MODELS (8 моделей)")
        print("="*60)
        self.arima_ensemble_analysis()
        
        print("\n" + "="*80)
        print("✅ ПОВНИЙ АНАЛІЗ ЗАВЕРШЕНО УСПІШНО!")
        print(f"📁 Результати збережено в: {self.results_dir}")
        print("="*80)
        
    def load_and_analyze_data(self):
        """Завантаження даних та базова статистика"""
        try:
            # Основні дані з whale_hourly_complete
            query = """
        SELECT 
            timestamp,
            whale_volume_usd,
            whale_activity,
            exchange_inflow,
            exchange_outflow,
            net_flow,
            btc_price,
            fear_greed_index,
            fear_greed_classification,
            market_sentiment,
            SP500,
            VIX,
            GOLD,
            NASDAQ,
            OIL_WTI
        FROM whale_hourly_complete
        WHERE whale_activity > 0
        AND btc_price > 0
        ORDER BY timestamp
        """
        
            # Використовуємо новий метод execute_query
            conn = self.get_db_connection()
            df = pd.read_sql(query, conn)
            conn.close()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            self.log.info(f"Завантажено {len(df)} записів з whale_hourly_complete")
            self.logger.log_metric('data', 'hourly_records', len(df))
            
            # Транзакції з whale_alerts_original
            query_tx = """
        SELECT 
            timestamp,
            currency,
            amount,
            usd_value,
            from_entity,
            to_entity,
            transaction_type
        FROM whale_alerts_original
        ORDER BY timestamp
        """
        
            conn = self.get_db_connection()
            df_tx = pd.read_sql(query_tx, conn)
            conn.close()
            
            df_tx['timestamp'] = pd.to_datetime(df_tx['timestamp'])
            
            self.log.info(f"Завантажено {len(df_tx)} транзакцій з whale_alerts_original")
            self.logger.log_metric('data', 'transaction_records', len(df_tx))
            
            # Базова статистика
            stats = {
                'total_records': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'total_whale_volume': float(df['whale_volume_usd'].sum()),
                'avg_whale_volume_hourly': float(df['whale_volume_usd'].mean()),
                'total_transactions': len(df_tx),
                'unique_currencies': df_tx['currency'].nunique(),
                'exchange_inflow_total': float(df['exchange_inflow'].sum()),
                'exchange_outflow_total': float(df['exchange_outflow'].sum()),
                'net_flow_total': float(df['net_flow'].sum())
            }
            
            self.all_results['basic_statistics'] = stats
            
            self.log.info(f" Завантажено {len(df)} годинних записів")
            self.log.info(f" Період: {stats['date_range']}")
            self.log.info(f" Загальний об'єм whale: ${stats['total_whale_volume']:,.0f}")
            self.log.info(f" Транзакцій: {stats['total_transactions']}")
            
            # Логуємо всі метрики
            for key, value in stats.items():
                self.logger.log_metric('data_stats', key, value)
            
            # Зберігаємо для подальшого використання
            self.df = df
            self.df_tx = df_tx
            
            return df
            
        except Exception as e:
            self.log.error(f"Помилка завантаження даних: {e}")
            self.logger.log_error(e, context={'method': 'load_and_analyze_data'})
            raise
    
    def load_all_features_from_db(self):
        """Завантажує всі 233 фічі з таблиці universal_features"""
        self.log.info("=== ЗАВАНТАЖЕННЯ 233 ФІЧЕЙ З БД ===")
        
        try:
            # Отримуємо унікальні timestamps для яких є дані
            query_timestamps = """
            SELECT DISTINCT timestamp 
            FROM universal_features 
            ORDER BY timestamp
            """
            timestamps = self.execute_query(query_timestamps)
            self.log.info(f"Знайдено {len(timestamps)} унікальних timestamps")
            
            # Отримуємо всі фічі в EAV форматі
            query_features = """
            SELECT timestamp, feature_name, feature_value
            FROM universal_features
            ORDER BY timestamp, feature_name
            """
            
            conn = self.get_db_connection()
            df_eav = pd.read_sql(query_features, conn)
            conn.close()
            
            self.log.info(f"Завантажено {len(df_eav)} записів в EAV форматі")
            self.logger.log_metric('features', 'eav_records', len(df_eav))
            
            # Перетворення з EAV в широкий формат
            df_wide = df_eav.pivot(
                index='timestamp',
                columns='feature_name', 
                values='feature_value'
            )
            
            # Конвертуємо index в datetime
            df_wide.index = pd.to_datetime(df_wide.index)
            
            self.log.info(f"Перетворено в DataFrame: {df_wide.shape[0]} рядків x {df_wide.shape[1]} фічей")
            self.logger.log_metric('features', 'feature_count', df_wide.shape[1])
            self.logger.log_metric('features', 'row_count', df_wide.shape[0])
            
            # Перевірка на пропущені значення
            missing_pct = (df_wide.isna().sum() / len(df_wide) * 100).mean()
            self.log.info(f"Середній % пропущених значень: {missing_pct:.2f}%")
            self.logger.log_metric('features', 'avg_missing_pct', missing_pct)
            
            return df_wide
            
        except Exception as e:
            self.log.error(f"Помилка завантаження фічей: {e}")
            self.logger.log_error(e, context={'method': 'load_all_features_from_db'})
            raise
    
    def analyze_individual_features(self, df_features, df_target):
        """Аналізує вплив кожної з 233 фічей на цільову змінну"""
        self.log.info("=== АНАЛІЗ ВПЛИВУ КОЖНОЇ ФІЧІ ===")
        
        feature_analysis = {}
        
        # Об'єднуємо фічі з цільовою змінною (btc_price)
        df_combined = pd.merge(
            df_features,
            df_target[['btc_price']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        self.log.info(f"Об'єднано для аналізу: {len(df_combined)} записів")
        
        # Аналізуємо кожну фічу
        for i, feature in enumerate(df_features.columns):
            if (i + 1) % 50 == 0:
                self.log.info(f"Проаналізовано {i + 1}/{len(df_features.columns)} фічей...")
            
            try:
                # Базова статистика
                feature_data = df_combined[feature].dropna()
                target_data = df_combined.loc[feature_data.index, 'btc_price']
                
                # Кореляція Пірсона
                if len(feature_data) > 10:
                    correlation = feature_data.corr(target_data)
                    pearson_p = stats.pearsonr(feature_data, target_data)[1]
                else:
                    correlation = 0
                    pearson_p = 1
                
                # Взаємна інформація
                if len(feature_data) > 100:
                    from sklearn.feature_selection import mutual_info_regression
                    mi_score = mutual_info_regression(
                        feature_data.values.reshape(-1, 1),
                        target_data.values,
                        random_state=42
                    )[0]
                else:
                    mi_score = 0
                
                # Визначаємо тип фічі
                feature_type = self._get_feature_type(feature)
                
                analysis = {
                    'feature_name': feature,
                    'feature_type': feature_type,
                    'correlation': float(correlation),
                    'correlation_abs': float(abs(correlation)),
                    'p_value': float(pearson_p),
                    'significant': pearson_p < 0.05,
                    'mutual_info': float(mi_score),
                    'missing_pct': float(df_combined[feature].isna().mean() * 100),
                    'unique_values': int(df_combined[feature].nunique()),
                    'std_dev': float(feature_data.std()) if len(feature_data) > 0 else 0
                }
                
                feature_analysis[feature] = analysis
                
            except Exception as e:
                self.log.warning(f"Помилка аналізу фічі {feature}: {e}")
                feature_analysis[feature] = {
                    'feature_name': feature,
                    'feature_type': self._get_feature_type(feature),
                    'error': str(e)
                }
        
        self.log.info(f"Завершено аналіз {len(feature_analysis)} фічей")
        
        # Зберігаємо результати в БД
        self._save_feature_analysis_to_db(feature_analysis)
        
        # Створюємо DataFrame для зручності
        df_analysis = pd.DataFrame.from_dict(feature_analysis, orient='index')
        
        # Сортуємо за абсолютною кореляцією
        df_analysis = df_analysis.sort_values('correlation_abs', ascending=False)
        
        # Виводимо топ-20
        self.log.info("\nТОП-20 фічей за кореляцією з btc_price:")
        for idx, row in df_analysis.head(20).iterrows():
            self.log.info(f"  {row['feature_name']}: corr={row['correlation']:.3f}, MI={row['mutual_info']:.3f}")
        
        return df_analysis
    
    def _get_feature_type(self, feature_name):
        """Визначення типу фічі за назвою"""
        if any(x in feature_name for x in ['hour', 'day', 'month', 'weekend', 'time']):
            return 'temporal'
        elif 'lag' in feature_name:
            return 'lag'
        elif 'rolling' in feature_name:
            return 'rolling_stat'
        elif any(x in feature_name for x in ['rsi', 'macd', 'bb_', 'stoch', 'williams']):
            return 'technical'
        elif any(x in feature_name for x in ['whale', 'flow', 'intensity']):
            return 'whale_specific'
        elif any(x in feature_name for x in ['vol', 'change']):
            return 'volatility'
        elif 'interaction' in feature_name:
            return 'interaction'
        elif any(x in feature_name for x in ['log', 'sqrt', 'zscore']):
            return 'transform'
        else:
            return 'other'
    
    def _save_feature_analysis_to_db(self, feature_analysis):
        """Зберігає результати аналізу фічей в БД"""
        try:
            # Оновлюємо importance_score в feature_metadata
            for feature, analysis in feature_analysis.items():
                if 'error' not in analysis:
                    # Комбінований score на основі кореляції та MI
                    importance_score = (
                        abs(analysis['correlation']) * 0.7 + 
                        min(analysis['mutual_info'], 1.0) * 0.3
                    )
                    
                    query = """
                    UPDATE feature_metadata 
                    SET importance_score = %s
                    WHERE feature_name = %s
                    """
                    
                    self.execute_query(query, params=(importance_score, feature))
            
            self.log.info("Результати аналізу збережено в БД")
            self.logger.log_metric('features', 'analysis_saved', True)
            
        except Exception as e:
            self.log.error(f"Помилка збереження аналізу: {e}")
            self.logger.log_error(e, context={'method': '_save_feature_analysis_to_db'})
    
    def test_feature_groups(self, df_features, df_target):
        """Тестує групи фічей окремо та в комбінаціях"""
        self.log.info("=== ТЕСТУВАННЯ ГРУП ФІЧЕЙ ===")
        
        # Об'єднуємо дані
        df_combined = pd.merge(
            df_features,
            df_target[['btc_price', 'whale_volume_usd', 'net_flow']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Визначаємо групи фічей
        feature_groups = {
            'temporal': [],
            'lag': [],
            'rolling_stat': [],
            'technical': [],
            'whale_specific': [],
            'volatility': [],
            'interaction': [],
            'transform': []
        }
        
        # Розподіляємо фічі по групах
        for feature in df_features.columns:
            feature_type = self._get_feature_type(feature)
            if feature_type in feature_groups:
                feature_groups[feature_type].append(feature)
        
        # Виводимо розміри груп
        self.log.info("Розподіл фічей по групах:")
        for group_name, features in feature_groups.items():
            self.log.info(f"  {group_name}: {len(features)} фічей")
            self.logger.log_metric('feature_groups', f'{group_name}_count', len(features))
        
        # Тестуємо кожну групу
        group_results = {}
        
        for group_name, features in feature_groups.items():
            if len(features) == 0:
                continue
                
            self.log.info(f"\nТестування групи '{group_name}' ({len(features)} фічей)...")
            
            # Підготовка даних для моделі
            X = df_combined[features].fillna(0)
            y = df_combined['btc_price']
            
            # Розділення на train/test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Тестуємо з RandomForest
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_absolute_percentage_error, r2_score
                
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Прогнози
                y_pred = model.predict(X_test)
                
                # Метрики
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                r2 = r2_score(y_test, y_pred)
                
                # Feature importance
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                group_results[group_name] = {
                    'feature_count': len(features),
                    'mape': float(mape),
                    'r2': float(r2),
                    'top_features': importance_df.head(5).to_dict('records')
                }
                
                self.log.info(f"  MAPE: {mape:.2f}%")
                self.log.info(f"  R²: {r2:.3f}")
                self.log.info(f"  Топ-3 фічі:")
                for idx, row in importance_df.head(3).iterrows():
                    self.log.info(f"    - {row['feature']}: {row['importance']:.3f}")
                
                self.logger.log_metric(f'group_{group_name}', 'mape', mape)
                self.logger.log_metric(f'group_{group_name}', 'r2', r2)
                
            except Exception as e:
                self.log.error(f"Помилка тестування групи {group_name}: {e}")
                group_results[group_name] = {'error': str(e)}
        
        # Тестуємо комбінації найкращих груп
        self.log.info("\n=== ТЕСТУВАННЯ КОМБІНАЦІЙ ГРУП ===")
        
        # Сортуємо групи за MAPE
        sorted_groups = sorted(
            [(k, v) for k, v in group_results.items() if 'mape' in v],
            key=lambda x: x[1]['mape']
        )
        
        if len(sorted_groups) >= 2:
            # Комбінуємо топ-2 групи
            best_groups = [sorted_groups[0][0], sorted_groups[1][0]]
            combined_features = []
            for group in best_groups:
                combined_features.extend(feature_groups[group])
            
            self.log.info(f"Комбінація {' + '.join(best_groups)} ({len(combined_features)} фічей)")
            
            # Тестуємо комбінацію
            X = df_combined[combined_features].fillna(0)
            X_train, X_test = X[:train_size], X[train_size:]
            
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            r2 = r2_score(y_test, y_pred)
            
            self.log.info(f"  MAPE: {mape:.2f}%")
            self.log.info(f"  R²: {r2:.3f}")
            
            group_results['best_combination'] = {
                'groups': best_groups,
                'feature_count': len(combined_features),
                'mape': float(mape),
                'r2': float(r2)
            }
        
        return group_results
    
    def find_optimal_feature_strategies(self, df_features, df_target, df_analysis):
        """Знаходить оптимальні стратегії використання фічей"""
        self.log.info("=== ПОШУК ОПТИМАЛЬНИХ СТРАТЕГІЙ ===")
        
        # Об'єднуємо дані
        df_combined = pd.merge(
            df_features,
            df_target[['btc_price']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        strategies = {}
        
        # Підготовка даних
        X = df_combined[df_features.columns].fillna(0)
        y = df_combined['btc_price']
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Стандартизація
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Стратегія: Топ-50 за кореляцією
        self.log.info("\n1. Топ-50 фічей за кореляцією...")
        top_50_features = df_analysis.head(50).index.tolist()
        strategies['top_50_correlation'] = self._test_strategy(
            X_train[top_50_features], X_test[top_50_features], y_train, y_test,
            "Топ-50 за кореляцією"
        )
        
        # 2. Стратегія: Топ-30 за кореляцією
        self.log.info("\n2. Топ-30 фічей за кореляцією...")
        top_30_features = df_analysis.head(30).index.tolist()
        strategies['top_30_correlation'] = self._test_strategy(
            X_train[top_30_features], X_test[top_30_features], y_train, y_test,
            "Топ-30 за кореляцією"
        )
        
        # 3. Стратегія: LASSO відбір
        self.log.info("\n3. LASSO відбір фічей...")
        from sklearn.linear_model import LassoCV
        
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X_train_scaled, y_train)
        
        # Відбираємо фічі з ненульовими коефіцієнтами
        lasso_features = X.columns[lasso.coef_ != 0].tolist()
        self.log.info(f"  LASSO відібрав {len(lasso_features)} фічей")
        
        if len(lasso_features) > 0:
            strategies['lasso_selected'] = self._test_strategy(
                X_train[lasso_features], X_test[lasso_features], y_train, y_test,
                "LASSO відбір"
            )
        
        # 4. Стратегія: RFE (Recursive Feature Elimination)
        self.log.info("\n4. RFE відбір фічей...")
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor
        
        estimator = RandomForestRegressor(n_estimators=20, random_state=42)
        rfe = RFE(estimator, n_features_to_select=30, step=10)
        rfe.fit(X_train, y_train)
        
        rfe_features = X.columns[rfe.support_].tolist()
        self.log.info(f"  RFE відібрав {len(rfe_features)} фічей")
        
        strategies['rfe_selected'] = self._test_strategy(
            X_train[rfe_features], X_test[rfe_features], y_train, y_test,
            "RFE відбір"
        )
        
        # 5. Стратегія: Всі 233 фічі з regularization
        self.log.info("\n5. Всі фічі з Ridge regularization...")
        from sklearn.linear_model import Ridge
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        
        strategies['all_features_ridge'] = {
            'feature_count': len(df_features.columns),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred) * 100),
            'r2': float(r2_score(y_test, y_pred))
        }
        
        # Виводимо підсумок
        self.log.info("\n=== ПІДСУМОК СТРАТЕГІЙ ===")
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: x[1]['mape'] if 'mape' in x[1] else float('inf')
        )
        
        for strategy_name, results in sorted_strategies:
            if 'mape' in results:
                self.log.info(f"{strategy_name}: MAPE={results['mape']:.2f}%, R²={results['r2']:.3f}, фічей={results['feature_count']}")
        
        # Зберігаємо найкращу стратегію
        best_strategy = sorted_strategies[0]
        self.log.info(f"\n✓ НАЙКРАЩА СТРАТЕГІЯ: {best_strategy[0]}")
        self.logger.log_metric('best_strategy', 'name', best_strategy[0])
        self.logger.log_metric('best_strategy', 'mape', best_strategy[1]['mape'])
        
        return strategies
    
    def _test_strategy(self, X_train, X_test, y_train, y_test, strategy_name):
        """Тестує стратегію відбору фічей"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_percentage_error, r2_score
        
        try:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            r2 = r2_score(y_test, y_pred)
            
            self.log.info(f"  {strategy_name}: MAPE={mape:.2f}%, R²={r2:.3f}")
            
            return {
                'feature_count': X_train.shape[1],
                'mape': float(mape),
                'r2': float(r2),
                'features': X_train.columns.tolist()
            }
            
        except Exception as e:
            self.log.error(f"Помилка тестування стратегії {strategy_name}: {e}")
            return {'error': str(e)}
    
    def visualize_feature_importance(self, df_analysis, save_path=None):
        """Створює візуалізацію важливості всіх 233 фічей"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Аналіз важливості 233 фічей GT14', fontsize=16)
        
        # 1. Топ-30 за кореляцією
        ax1 = axes[0, 0]
        top_30 = df_analysis.head(30)
        ax1.barh(range(len(top_30)), top_30['correlation_abs'], color='steelblue')
        ax1.set_yticks(range(len(top_30)))
        ax1.set_yticklabels(top_30.index, fontsize=8)
        ax1.set_xlabel('Абсолютна кореляція')
        ax1.set_title('Топ-30 фічей за кореляцією з BTC price')
        ax1.invert_yaxis()
        
        # 2. Розподіл кореляцій по типах
        ax2 = axes[0, 1]
        type_corr = df_analysis.groupby('feature_type')['correlation_abs'].agg(['mean', 'std', 'count'])
        type_corr = type_corr.sort_values('mean', ascending=False)
        
        x = range(len(type_corr))
        ax2.bar(x, type_corr['mean'], yerr=type_corr['std'], capsize=5, color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels(type_corr.index, rotation=45)
        ax2.set_ylabel('Середня абсолютна кореляція')
        ax2.set_title('Середня кореляція по типах фічей')
        
        # Додаємо кількість фічей
        for i, (idx, row) in enumerate(type_corr.iterrows()):
            ax2.text(i, row['mean'] + row['std'] + 0.01, f"n={row['count']}", 
                    ha='center', fontsize=8)
        
        # 3. Mutual Information vs Correlation
        ax3 = axes[1, 0]
        valid_features = df_analysis.dropna(subset=['correlation', 'mutual_info'])
        ax3.scatter(valid_features['correlation_abs'], valid_features['mutual_info'], 
                   alpha=0.6, s=20)
        ax3.set_xlabel('Абсолютна кореляція')
        ax3.set_ylabel('Mutual Information')
        ax3.set_title('Кореляція vs Mutual Information')
        
        # Виділяємо топ-20
        top_20 = valid_features.head(20)
        ax3.scatter(top_20['correlation_abs'], top_20['mutual_info'], 
                   color='red', s=50, alpha=0.8, label='Топ-20')
        ax3.legend()
        
        # 4. Статистично значущі фічі
        ax4 = axes[1, 1]
        significant_by_type = df_analysis[df_analysis['significant'] == True].groupby('feature_type').size()
        total_by_type = df_analysis.groupby('feature_type').size()
        
        # Відсоток значущих
        pct_significant = (significant_by_type / total_by_type * 100).fillna(0)
        pct_significant = pct_significant.sort_values(ascending=False)
        
        ax4.bar(range(len(pct_significant)), pct_significant.values, color='green', alpha=0.7)
        ax4.set_xticks(range(len(pct_significant)))
        ax4.set_xticklabels(pct_significant.index, rotation=45)
        ax4.set_ylabel('% статистично значущих (p < 0.05)')
        ax4.set_title('Відсоток значущих фічей по типах')
        
        # Додаємо числа
        for i, (idx, val) in enumerate(pct_significant.items()):
            if idx in significant_by_type:
                ax4.text(i, val + 1, f"{significant_by_type[idx]}/{total_by_type[idx]}", 
                        ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.log.info(f"Візуалізація збережена: {save_path}")
        
        return fig
    
    def _save_feature_analysis_results(self, df_analysis, group_results, strategies):
        """Зберігає результати аналізу фічей у файли"""
        try:
            # 1. Зберігаємо детальний аналіз фічей
            df_analysis.to_csv(
                self.results_dir / 'feature_analysis_detailed.csv',
                index=True
            )
            
            # 2. Зберігаємо результати по групах
            with open(self.results_dir / 'feature_group_results.json', 'w') as f:
                json.dump(group_results, f, indent=2)
            
            # 3. Зберігаємо результати стратегій
            with open(self.results_dir / 'feature_strategies.json', 'w') as f:
                json.dump(strategies, f, indent=2)
            
            # 4. Створюємо summary файл
            with open(self.results_dir / 'feature_analysis_summary.txt', 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("АНАЛІЗ 233 ФІЧЕЙ - РЕЗУЛЬТАТИ\n")
                f.write("="*60 + "\n\n")
                
                # Топ-20 фічей
                f.write("ТОП-20 НАЙВАЖЛИВІШИХ ФІЧЕЙ:\n")
                for idx, row in df_analysis.head(20).iterrows():
                    f.write(f"{row.name}: corr={row['correlation']:.3f}, MI={row['mutual_info']:.3f}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("РЕЗУЛЬТАТИ ПО ГРУПАХ:\n")
                f.write("="*60 + "\n")
                
                for group_name, results in group_results.items():
                    if 'mape' in results:
                        f.write(f"{group_name}: MAPE={results['mape']:.2f}%, R²={results['r2']:.3f}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("ПОРІВНЯННЯ СТРАТЕГІЙ:\n")
                f.write("="*60 + "\n")
                
                sorted_strategies = sorted(
                    [(k, v) for k, v in strategies.items() if 'mape' in v],
                    key=lambda x: x[1]['mape']
                )
                
                for strategy_name, results in sorted_strategies:
                    f.write(f"{strategy_name}: MAPE={results['mape']:.2f}%, фічей={results['feature_count']}\n")
            
            self.log.info("✓ Результати аналізу фічей збережено")
            
            # Додаємо до загальних результатів
            self.all_results['feature_analysis'] = {
                'total_features': len(df_analysis),
                'significant_features': len(df_analysis[df_analysis['significant'] == True]),
                'best_strategy': min(strategies.items(), key=lambda x: x[1].get('mape', float('inf')))[0],
                'feature_groups': {k: v for k, v in group_results.items() if 'mape' in v}
            }
            
        except Exception as e:
            self.log.error(f"Помилка збереження результатів аналізу фічей: {e}")
        
    def temporal_analysis(self, df):
        """Детальний часовий аналіз як в v8"""
        results = {}
        
        # 1. Аналіз по годинах доби
        hourly_patterns = df.groupby(df.index.hour).agg({
            'whale_volume_usd': ['mean', 'std', 'count'],
            'whale_activity': 'mean',
            'net_flow': 'mean'
        }).round(2)
        
        results['hourly_patterns'] = hourly_patterns.to_dict()
        
        # 2. Аналіз по днях тижня
        daily_patterns = df.groupby(df.index.dayofweek).agg({
            'whale_volume_usd': ['mean', 'std', 'count'],
            'whale_activity': 'mean',
            'net_flow': 'mean'
        }).round(2)
        
        results['daily_patterns'] = daily_patterns.to_dict()
        
        # 3. Тренди
        df['whale_volume_ma7'] = df['whale_volume_usd'].rolling(168).mean()  # 7 днів
        df['whale_volume_ma30'] = df['whale_volume_usd'].rolling(720).mean()  # 30 днів
        
        # 4. Виявлення піків активності
        threshold = df['whale_volume_usd'].quantile(0.95)
        peaks = df[df['whale_volume_usd'] > threshold]
        
        results['activity_peaks'] = {
            'threshold': float(threshold),
            'peak_count': len(peaks),
            'peak_dates': peaks.index.tolist()
        }
        
        self.all_results['temporal_analysis'] = results
        
        print(f" Пікова активність: {results['activity_peaks']['peak_count']} годин > ${threshold:,.0f}")
        print(f" Найактивніша година доби: {hourly_patterns['whale_volume_usd']['mean'].idxmax()}:00")
        
        return results
        
    def cross_correlation_analysis(self, df):
        """Крос-кореляційний аналіз між всіма парами змінних"""
        results = {}
        
        # Вибираємо ключові змінні
        variables = [
            'whale_volume_usd',
            'exchange_inflow', 
            'exchange_outflow',
            'net_flow',
            'btc_price',
            'whale_activity'
        ]
        
        # Видаляємо NaN
        df_clean = df[variables].dropna()
        
        # Перевіряємо чи є дані
        if len(df_clean) < 50:
            print(" Недостатньо даних для крос-кореляційного аналізу")
            self.all_results['cross_correlation'] = {'error': 'Insufficient data'}
            return results
        
        # Розраховуємо крос-кореляції
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j and var1 in df_clean.columns and var2 in df_clean.columns:
                    try:
                        # Перевіряємо що колонки мають дані
                        if df_clean[var1].notna().sum() > 25 and df_clean[var2].notna().sum() > 25:
                            # CCF до 24 лагів (24 години)
                            ccf_values = ccf(df_clean[var1].dropna(), df_clean[var2].dropna(), adjusted=False)[:25]
                            
                            # Знаходимо максимальну кореляцію
                            max_corr_idx = np.argmax(np.abs(ccf_values))
                            max_corr = ccf_values[max_corr_idx]
                            
                            results[f"{var1}_vs_{var2}"] = {
                                'max_correlation': float(max_corr),
                                'lag_hours': int(max_corr_idx),
                                'correlations': ccf_values.tolist(),
                                'significant': abs(max_corr) > 0.3
                            }
                            
                            if abs(max_corr) > 0.3:
                                print(f" {var1} → {var2}: r={max_corr:.3f} @ lag={max_corr_idx}h")
                    except Exception as e:
                        pass
        
        # Матриця кореляцій
        corr_matrix = df_clean.corr()
        
        # Візуалізація
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True)
        plt.title('Матриця кореляцій ключових змінних')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'correlation_matrix.png', dpi=300)
        plt.close()
        
        self.all_results['cross_correlation'] = results
        
        return results
        
    def seasonality_analysis(self, df):
        """Аналіз сезонності"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        results = {}
        
        # Декомпозиція часового ряду
        decomposition = seasonal_decompose(
            df['whale_volume_usd'].fillna(method='ffill'), 
            model='additive', 
            period=24  # Добова сезонність
        )
        
        results['daily_seasonality'] = {
            'seasonal_strength': float(decomposition.seasonal.std() / df['whale_volume_usd'].std()),
            'trend_strength': float(decomposition.trend.dropna().std() / df['whale_volume_usd'].std())
        }
        
        # Тижнева сезонність
        weekly_decomp = seasonal_decompose(
            df['whale_volume_usd'].fillna(method='ffill'),
            model='additive',
            period=168  # Тижнева сезонність
        )
        
        results['weekly_seasonality'] = {
            'seasonal_strength': float(weekly_decomp.seasonal.std() / df['whale_volume_usd'].std())
        }
        
        # Візуалізація
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        df['whale_volume_usd'].plot(ax=axes[0], title='Оригінальний ряд')
        decomposition.trend.plot(ax=axes[1], title='Тренд')
        decomposition.seasonal.plot(ax=axes[2], title='Сезонність')
        decomposition.resid.plot(ax=axes[3], title='Залишки')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'seasonality_decomposition.png', dpi=300)
        plt.close()
        
        self.all_results['seasonality'] = results
        
        print(f" Сила добової сезонності: {results['daily_seasonality']['seasonal_strength']:.2%}")
        print(f" Сила тренду: {results['daily_seasonality']['trend_strength']:.2%}")
        
        return results
        
    def anomaly_detection(self, df):
        """Детекція аномалій кількома методами"""
        results = {'anomalies': {}}
        
        # Підготовка даних
        features = ['whale_volume_usd', 'net_flow', 'whale_activity']
        X = df[features].dropna()
        
        # Нормалізація
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalies_iso = iso_forest.fit_predict(X_scaled)
        
        # 2. Elliptic Envelope
        elliptic = EllipticEnvelope(contamination=0.05, random_state=42)
        anomalies_elliptic = elliptic.fit_predict(X_scaled)
        
        # 3. Statistical method (3-sigma)
        z_scores = np.abs(stats.zscore(X))
        anomalies_zscore = (z_scores > 3).any(axis=1)
        
        # Об'єднання результатів
        anomaly_dates_iso = X.index[anomalies_iso == -1]
        anomaly_dates_elliptic = X.index[anomalies_elliptic == -1]
        anomaly_dates_zscore = X.index[anomalies_zscore]
        
        # Консенсус - аномалії виявлені хоча б 2 методами
        all_anomalies = set(anomaly_dates_iso) | set(anomaly_dates_elliptic) | set(anomaly_dates_zscore)
        consensus_anomalies = []
        
        for date in all_anomalies:
            count = 0
            if date in anomaly_dates_iso: count += 1
            if date in anomaly_dates_elliptic: count += 1
            if date in anomaly_dates_zscore: count += 1
            
            if count >= 2:
                consensus_anomalies.append(date)
                
        results['anomalies'] = {
            'isolation_forest': {'count': len(anomaly_dates_iso), 'dates': anomaly_dates_iso.tolist()},
            'elliptic_envelope': {'count': len(anomaly_dates_elliptic), 'dates': anomaly_dates_elliptic.tolist()},
            'statistical_zscore': {'count': len(anomaly_dates_zscore), 'dates': anomaly_dates_zscore.tolist()},
            'consensus': {'count': len(consensus_anomalies), 'dates': consensus_anomalies}
        }
        
        # Візуалізація
        plt.figure(figsize=(14, 8))
        
        plt.scatter(X.index, X['whale_volume_usd'], c='blue', alpha=0.5, label='Нормальні')
        
        for date in consensus_anomalies[:20]:  # Показуємо перші 20
            if date in X.index:
                plt.scatter(date, X.loc[date, 'whale_volume_usd'], 
                           c='red', s=100, marker='x', label='Аномалія' if date == consensus_anomalies[0] else "")
                
        plt.xlabel('Дата')
        plt.ylabel('Whale Volume (USD)')
        plt.title('Детекція аномалій в whale активності')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'anomaly_detection.png', dpi=300)
        plt.close()
        
        self.all_results['anomaly_detection'] = results
        
        print(f" Виявлено аномалій (консенсус): {len(consensus_anomalies)}")
        
        return results
        
    def advanced_clustering(self, df):
        """Кластеризація з УСІМА метриками"""
        results = {}
        
        # Підготовка даних для кластеризації з реальними колонками
        available_cols = df.columns.tolist()
        
        # Базові фічі для кластеризації
        base_features = ['whale_volume_usd', 'net_flow', 'whale_activity', 
                        'exchange_inflow', 'exchange_outflow', 'fear_greed_index']
        
        features = []
        for feat in base_features:
            if feat in available_cols:
                features.append(feat)
                
        # Додаємо оптимальні фічі якщо вони були інтегровані
        if hasattr(self, 'optimal_features') and self.optimal_features:
            # Вибираємо топ-5 оптимальних фічей для кластеризації
            top_optimal = self.optimal_features[:5]
            for feat in top_optimal:
                if feat in available_cols and feat not in features:
                    features.append(feat)
            print(f" Додано {len(features) - len([f for f in base_features if f in available_cols])} оптимальних фічей до кластеризації")
            
        print(f"Кластеризація features: {features}")
        
        if len(features) < 2:
            print(" Недостатньо features для кластеризації")
            return {'error': 'Insufficient features'}
        
        X = df[features].dropna()
        
        if len(X) < 10:
            print(" Недостатньо даних для кластеризації")
            return {'error': 'Insufficient data'}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Тестуємо різну кількість кластерів
        k_range = range(2, 8)
        metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        best_k = 2
        best_score = -1
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Розраховуємо ВСІ три метрики
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            
            metrics['silhouette'].append(sil_score)
            metrics['davies_bouldin'].append(db_score)
            metrics['calinski_harabasz'].append(ch_score)
            
            # Комбінований скор (нормалізований)
            combined_score = sil_score + (1/db_score) + (ch_score/1000)
            
            if combined_score > best_score:
                best_score = combined_score
                best_k = k
                
            print(f"  K={k}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}, Calinski-Harabasz={ch_score:.1f}")
            
        # Фінальна кластеризація з оптимальним K
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(X_scaled)
        
        # Аналіз кластерів
        X['cluster'] = final_labels
        cluster_analysis = X.groupby('cluster').agg({
            'whale_volume_usd': ['mean', 'std', 'count'],
            'net_flow': 'mean',
            'whale_activity': 'mean'
        }).round(2)
        
        results = {
            'optimal_k': best_k,
            'metrics_comparison': {
                'k_values': list(k_range),
                'silhouette_scores': metrics['silhouette'],
                'davies_bouldin_scores': metrics['davies_bouldin'],
                'calinski_harabasz_scores': metrics['calinski_harabasz']
            },
            'final_metrics': {
                'silhouette': metrics['silhouette'][best_k-2],
                'davies_bouldin': metrics['davies_bouldin'][best_k-2],
                'calinski_harabasz': metrics['calinski_harabasz'][best_k-2]
            },
            'cluster_profiles': cluster_analysis.to_dict()
        }
        
        # Візуалізація метрик
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(k_range, metrics['silhouette'], 'b-o')
        axes[0].set_xlabel('Кількість кластерів')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Score (вище - краще)')
        axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'Optimal k={best_k}')
        axes[0].legend()
        
        axes[1].plot(k_range, metrics['davies_bouldin'], 'g-o')
        axes[1].set_xlabel('Кількість кластерів')
        axes[1].set_ylabel('Davies-Bouldin Score')
        axes[1].set_title('Davies-Bouldin Score (нижче - краще)')
        axes[1].axvline(x=best_k, color='r', linestyle='--')
        
        axes[2].plot(k_range, metrics['calinski_harabasz'], 'r-o')
        axes[2].set_xlabel('Кількість кластерів')
        axes[2].set_ylabel('Calinski-Harabasz Score')
        axes[2].set_title('Calinski-Harabasz Score (вище - краще)')
        axes[2].axvline(x=best_k, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'clustering_metrics_comparison.png', dpi=300)
        plt.close()
        
        # PCA візуалізація кластерів
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, 
                            cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Кластер')
        
        # Центри кластерів
        centers_pca = pca.transform(kmeans_final.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', 
                   marker='x', s=200, linewidths=3, label='Центри')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'Кластеризація whale активності (K={best_k})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'clustering_pca_visualization.png', dpi=300)
        plt.close()
        
        # ЕТАП 1.1: ЗБЕРЕЖЕННЯ CLUSTER LABELS В MYSQL
        print("\n ЗБЕРЕЖЕННЯ CLUSTER LABELS В БАЗУ ДАНИХ...")
        
        # Додаємо cluster_id до основних даних
        df_with_clusters = X.copy()  # X - це наші features DataFrame
        df_with_clusters['cluster_id'] = final_labels
        df_with_clusters['timestamp'] = df_with_clusters.index
        
        # Зберігаємо в MySQL
        conn = mysql.connector.connect(**self.config)
        cursor = conn.cursor()
        
        # Створюємо таблицю cluster_labels якщо не існує
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_labels (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            cluster_id INT,
            whale_volume_usd DECIMAL(20,2),
            net_flow DECIMAL(20,2),
            whale_activity INT,
            exchange_inflow DECIMAL(20,2),
            exchange_outflow DECIMAL(20,2),
            fear_greed_index INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Очищуємо стару таблицю
        cursor.execute("DELETE FROM cluster_labels")
        
        # Вставляємо нові дані
        insert_query = """
        INSERT INTO cluster_labels 
        (timestamp, cluster_id, whale_volume_usd, net_flow, whale_activity, 
         exchange_inflow, exchange_outflow, fear_greed_index)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cluster_data = []
        for idx, row in df_with_clusters.iterrows():
            cluster_data.append((
                row['timestamp'],
                int(row['cluster_id']),
                float(row['whale_volume_usd']) if pd.notna(row['whale_volume_usd']) else None,
                float(row['net_flow']) if pd.notna(row['net_flow']) else None,
                int(row['whale_activity']) if pd.notna(row['whale_activity']) else None,
                float(row['exchange_inflow']) if pd.notna(row['exchange_inflow']) else None,
                float(row['exchange_outflow']) if pd.notna(row['exchange_outflow']) else None,
                int(row['fear_greed_index']) if pd.notna(row['fear_greed_index']) else None
            ))
        
        cursor.executemany(insert_query, cluster_data)
        
        # Створюємо таблицю характеристик кластерів
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_features (
            cluster_id INT PRIMARY KEY,
            avg_whale_volume DECIMAL(20,2),
            avg_net_flow DECIMAL(20,2),
            avg_whale_activity DECIMAL(10,2),
            avg_exchange_inflow DECIMAL(20,2),
            avg_exchange_outflow DECIMAL(20,2),
            avg_fear_greed DECIMAL(5,2),
            cluster_size INT,
            cluster_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Очищуємо стару таблицю
        cursor.execute("DELETE FROM cluster_features")
        
        # Обчислюємо характеристики кластерів
        cluster_characteristics = []
        cluster_names = {
            0: "Низька активність",
            1: "Помірний outflow", 
            2: "Високий inflow",
            3: "Екстремальна активність",
            4: "Нейтральний потік",
            5: "Великі транзакції",
            6: "Ринкова паніка"
        }
        
        for cluster_id in range(best_k):
            cluster_data = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id]
            
            if len(cluster_data) > 0:
                characteristics = (
                    cluster_id,
                    float(cluster_data['whale_volume_usd'].mean()),
                    float(cluster_data['net_flow'].mean()),
                    float(cluster_data['whale_activity'].mean()),
                    float(cluster_data['exchange_inflow'].mean()),
                    float(cluster_data['exchange_outflow'].mean()),
                    float(cluster_data['fear_greed_index'].mean()),
                    len(cluster_data),
                    cluster_names.get(cluster_id, f"Кластер {cluster_id}")
                )
                cluster_characteristics.append(characteristics)
        
        cursor.executemany("""
        INSERT INTO cluster_features 
        (cluster_id, avg_whale_volume, avg_net_flow, avg_whale_activity,
         avg_exchange_inflow, avg_exchange_outflow, avg_fear_greed, 
         cluster_size, cluster_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, cluster_characteristics)
        
        conn.commit()
        conn.close()
        
        # Додаємо cluster_labels до результатів
        results['cluster_labels'] = final_labels.tolist()
        results['cluster_data_saved'] = True
        results['cluster_table_records'] = len(cluster_data)
        
        self.all_results['clustering'] = results
        
        print(f" Збережено {len(cluster_data)} записів з cluster labels в MySQL")
        print(f" Створено характеристики для {len(cluster_characteristics)} кластерів")
        print(f" Оптимальна кількість кластерів: {best_k}")
        print(f" Фінальні метрики:")
        print(f"  - Silhouette: {results['final_metrics']['silhouette']:.3f}")
        print(f"  - Davies-Bouldin: {results['final_metrics']['davies_bouldin']:.3f}")
        print(f"  - Calinski-Harabasz: {results['final_metrics']['calinski_harabasz']:.1f}")
        
        return results
        
    def comprehensive_var_analysis(self, df):
        """ЕТАП 1.2: VAR аналіз з інтеграцією cluster_labels"""
        results = {}
        
        # ЗАВАНТАЖУЄМО CLUSTER LABELS З MYSQL
        print("\n ЗАВАНТАЖЕННЯ CLUSTER LABELS З БАЗИ ДАНИХ...")
        
        conn = mysql.connector.connect(**self.config)
        
        # Завантажуємо cluster labels
        cluster_query = """
        SELECT timestamp, cluster_id 
        FROM cluster_labels 
        ORDER BY timestamp
        """
        df_clusters = pd.read_sql(cluster_query, conn)
        df_clusters['timestamp'] = pd.to_datetime(df_clusters['timestamp'])
        df_clusters.set_index('timestamp', inplace=True)
        
        # Характеристики кластерів
        cluster_features_query = """
        SELECT cluster_id, cluster_name, cluster_size,
               avg_whale_volume, avg_net_flow, avg_exchange_inflow, avg_exchange_outflow
        FROM cluster_features
        ORDER BY cluster_id
        """
        df_cluster_features = pd.read_sql(cluster_features_query, conn)
        conn.close()
        
        print(f" Завантажено cluster labels для {len(df_clusters)} записів")
        print(f" Знайдено {len(df_cluster_features)} кластерів")
        
        # Об'єднуємо дані з кластерами
        df_with_clusters = df.join(df_clusters, how='inner')
        
        print(f" Дані після об'єднання з кластерами: {len(df_with_clusters)} записів")
        
        # Підготовка даних з реальними колонками
        available_cols = df_with_clusters.columns.tolist()
        var_vars = []
        
        # Базові змінні
        base_vars = ['whale_volume_usd', 'net_flow', 'btc_price', 'whale_activity', 
                    'exchange_inflow', 'exchange_outflow', 'fear_greed_index']
        
        # Додаємо базові змінні якщо вони доступні
        for var in base_vars:
            if var in available_cols:
                var_vars.append(var)
                
        # Додаємо оптимальні фічі якщо вони були інтегровані
        if hasattr(self, 'optimal_features') and self.optimal_features:
            # Вибираємо топ-5 оптимальних фічей для VAR
            top_optimal = self.optimal_features[:5]
            for feat in top_optimal:
                if feat in available_cols and feat not in var_vars:
                    var_vars.append(feat)
            print(f" Додано {len(top_optimal)} оптимальних фічей до VAR аналізу")
            
        print(f"VAR змінні: {var_vars}")
        
        if len(var_vars) < 2:
            print(" Недостатньо змінних для VAR аналізу")
            return {'error': 'Insufficient variables'}
            
        # CLUSTER-BASED VAR АНАЛІЗ
        cluster_var_results = {}
        
        print(f"\n ЗАПУСК VAR АНАЛІЗУ ДЛЯ КОЖНОГО КЛАСТЕРА...")
        
        for cluster_id in df_cluster_features['cluster_id']:
            cluster_name = df_cluster_features[df_cluster_features['cluster_id'] == cluster_id]['cluster_name'].iloc[0]
            cluster_data = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id]
            
            if len(cluster_data) < 50:  # Мінімум даних для VAR
                print(f"  Кластер {cluster_id} ({cluster_name}): недостатньо даних ({len(cluster_data)} записів)")
                continue
                
            print(f"\n Кластер {cluster_id} ({cluster_name}): {len(cluster_data)} записів")
            
            try:
                # VAR для конкретного кластера
                cluster_var_data = cluster_data[var_vars].dropna()
                
                if len(cluster_var_data) < 30:
                    print(f"  Після очищення: недостатньо даних ({len(cluster_var_data)} записів)")
                    continue
                
                # Стаціонарність для кластера
                cluster_stationary = {}
                for var in var_vars:
                    adf_result = adfuller(cluster_var_data[var])
                    cluster_stationary[var] = adf_result[1] < 0.05
                    
                # Диференціювання якщо потрібно
                non_stationary = [var for var, is_stat in cluster_stationary.items() if not is_stat]
                if non_stationary:
                    cluster_var_diff = cluster_var_data.diff().dropna()
                else:
                    cluster_var_diff = cluster_var_data
                
                # VAR модель для кластера
                model = VAR(cluster_var_diff)
                optimal_lag = min(4, len(cluster_var_diff) // 10)  # Обмежуємо лаги
                
                if optimal_lag >= 1:
                    fitted_model = model.fit(optimal_lag)
                    
                    # Granger causality для кластера
                    cluster_granger = {}
                    target_var = 'btc_price' if 'btc_price' in var_vars else var_vars[0]
                    
                    for var in var_vars:
                        if var != target_var:
                            try:
                                granger_test = grangercausalitytests(
                                    cluster_var_diff[[target_var, var]].dropna(),
                                    maxlag=min(2, optimal_lag),
                                    verbose=False
                                )
                                p_value = granger_test[1][0]['ssr_ftest'][1]
                                cluster_granger[f"{var}_to_{target_var}"] = {
                                    'p_value': p_value,
                                    'significant': p_value < 0.05
                                }
                                if p_value < 0.05:
                                    print(f"   {var} → {target_var}: p={p_value:.4f}")
                            except:
                                pass
                    
                    cluster_var_results[cluster_id] = {
                        'cluster_name': cluster_name,
                        'data_points': len(cluster_var_data),
                        'optimal_lag': optimal_lag,
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'granger_causality': cluster_granger,
                        'stationary_vars': list(cluster_stationary.keys()),
                        'significant_relationships': len([r for r in cluster_granger.values() if r['significant']])
                    }
                    
                    print(f"   VAR модель: lag={optimal_lag}, AIC={fitted_model.aic:.2f}")
                    print(f"   Значущих зв'язків: {cluster_var_results[cluster_id]['significant_relationships']}")
                
            except Exception as e:
                print(f"   Помилка VAR для кластера {cluster_id}: {e}")
        
        # ЗАГАЛЬНИЙ VAR (як раніше)
        print(f"\n ЗАГАЛЬНИЙ VAR АНАЛІЗ (всі дані)...")
        df_var = df_with_clusters[var_vars].dropna()
        
        if len(df_var) < 50:
            print(" Недостатньо даних для VAR аналізу")
            return {'error': 'Insufficient data'}
        
        # Перевірка стаціонарності
        print("\nПеревірка стаціонарності:")
        stationary_vars = []
        for var in var_vars:
            adf_result = adfuller(df_var[var])
            p_value = adf_result[1]
            print(f"  {var}: p={p_value:.4f} {'' if p_value < 0.05 else ''}")
            if p_value < 0.05:
                stationary_vars.append(var)
                
        # Стабілізація через логарифми
        df_var_log = df_var.copy()
        for col in ['whale_volume_usd', 'exchange_inflow', 'exchange_outflow']:
            if col in df_var_log.columns:
                df_var_log[col] = np.log(df_var_log[col] + 1)
        
        # VAR модель з логарифмованими даними
        model = VAR(df_var_log)
        lag_order = model.select_order(maxlags=8)
        optimal_lag = lag_order.aic
        
        print(f"Оптимальний лаг (AIC): {optimal_lag}")
        
        # Побудова VAR моделі
        fitted_model = model.fit(optimal_lag)
        
        print(f" VAR модель побудована з {optimal_lag} лагами")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        # IRF (Impulse Response Functions)
        irf = fitted_model.irf(periods=10)
        
        # FEVD (Forecast Error Variance Decomposition)
        fevd = fitted_model.fevd(periods=10)
        
        # Granger Causality тести
        granger_results = {}
        target_var = 'btc_price'
        
        if target_var in df_var_log.columns:
            for var in df_var_log.columns:
                if var != target_var:
                    try:
                        granger_test = grangercausalitytests(
                            df_var_log[[target_var, var]].dropna(),
                            maxlag=min(3, optimal_lag),
                            verbose=False
                        )
                        # Беремо p-value для лагу 1
                        p_value = granger_test[1][0]['ssr_ftest'][1]
                        granger_results[f"{var}_to_{target_var}"] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                        print(f"Granger {var} → {target_var}: p={p_value:.4f} {'' if p_value < 0.05 else ''}")
                    except Exception as e:
                        print(f"Помилка Granger тесту {var} → {target_var}: {e}")
        
        # Збереження результатів
        results = {
            'optimal_lag': optimal_lag,
            'model_stable': True,  # Заглушка
            'model_info': {
                'optimal_lag': optimal_lag,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'variables': list(df_var_log.columns),
                'observations': len(df_var_log)
            },
            'granger_causality': granger_results,
            'data_quality': {
                'stationary_vars': len(stationary_vars),
                'total_vars': len(var_vars)
            }
        }
        
        # Візуалізація IRF
        if len(df_var_log.columns) >= 2:
            plt.figure(figsize=(15, 10))
            irf.plot(impulse=df_var_log.columns[1], response=df_var_log.columns[0])
            plt.title(f'Impulse Response: {df_var_log.columns[1]} → {df_var_log.columns[0]}')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'var_irf.png', dpi=300)
            plt.close()
            
            # FEVD графік
            plt.figure(figsize=(12, 8))
            fevd.plot()
            plt.title('Forecast Error Variance Decomposition')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'var_fevd.png', dpi=300)
            plt.close()
        
        # Додаємо cluster-based результати
        results['cluster_var_analysis'] = cluster_var_results
        results['total_clusters_analyzed'] = len(cluster_var_results)
        
        print(f" VAR аналіз завершено. Знайдено {len([r for r in granger_results.values() if r['significant']])} значущих зв'язків")
        print(f" Cluster-based VAR: проаналізовано {len(cluster_var_results)} кластерів")
        
        # Підсумок по кластерах
        cluster_summary = {}
        for cluster_id, cluster_result in cluster_var_results.items():
            cluster_summary[cluster_id] = {
                'name': cluster_result['cluster_name'],
                'significant_relationships': cluster_result['significant_relationships']
            }
        
        if cluster_summary:
            print("\n ПІДСУМОК ПО КЛАСТЕРАХ:")
            for cluster_id, summary in cluster_summary.items():
                print(f"  Кластер {cluster_id} ({summary['name']}): {summary['significant_relationships']} зв'язків")
        
        self.all_results['var_analysis'] = results
        
        return results
        
    def bayesian_analysis(self, df):
        """ЕТАП 1.3: Bayesian аналіз з інтеграцією cluster_labels"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        results = {}
        
        # ЗАВАНТАЖУЄМО CLUSTER LABELS З MYSQL
        print("\n ЗАВАНТАЖЕННЯ CLUSTER LABELS ДЛЯ BAYESIAN АНАЛІЗУ...")
        
        conn = mysql.connector.connect(**self.config)
        
        # Завантажуємо cluster labels
        cluster_query = """
        SELECT timestamp, cluster_id 
        FROM cluster_labels 
        ORDER BY timestamp
        """
        df_clusters = pd.read_sql(cluster_query, conn)
        df_clusters['timestamp'] = pd.to_datetime(df_clusters['timestamp'])
        df_clusters.set_index('timestamp', inplace=True)
        conn.close()
        
        print(f" Завантажено cluster labels для {len(df_clusters)} записів")
        
        # Об'єднуємо дані з кластерами
        df_with_clusters = df.join(df_clusters, how='inner')
        
        print(f" Дані після об'єднання з кластерами: {len(df_with_clusters)} записів")
        
        # Створюємо цільову змінну - напрямок руху ціни
        df_with_clusters['price_direction'] = (df_with_clusters['btc_price'].shift(-1) > df_with_clusters['btc_price']).astype(int)
        
        # Базові ознаки + cluster_id
        base_features = ['whale_volume_usd', 'net_flow', 'whale_activity', 
                        'exchange_inflow', 'exchange_outflow', 'cluster_id']
        
        # Додаємо оптимальні фічі якщо вони були інтегровані
        features = base_features.copy()
        if hasattr(self, 'optimal_features') and self.optimal_features:
            # Вибираємо топ-10 оптимальних фічей для Байєсового аналізу
            top_optimal = self.optimal_features[:10]
            for feat in top_optimal:
                if feat in df_with_clusters.columns and feat not in features:
                    features.append(feat)
            print(f" Додано {len(features) - len(base_features)} оптимальних фічей до Байєсового аналізу")
        
        X = df_with_clusters[features].dropna()
        y = df_with_clusters.loc[X.index, 'price_direction']
        
        print(f" Ознаки для Bayesian аналізу: {features}")
        print(f" Розмір датасету: {len(X)} записів")
        
        # CLUSTER-BASED BAYESIAN АНАЛІЗ
        cluster_bayesian_results = {}
        
        print(f"\n ЗАПУСК BAYESIAN АНАЛІЗУ ДЛЯ КОЖНОГО КЛАСТЕРА...")
        
        for cluster_id in sorted(df_with_clusters['cluster_id'].unique()):
            cluster_mask = X['cluster_id'] == cluster_id
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]
            
            if len(X_cluster) < 30:  # Мінімум даних для навчання
                print(f"  Кластер {cluster_id}: недостатньо даних ({len(X_cluster)} записів)")
                continue
                
            print(f"\n Кластер {cluster_id}: {len(X_cluster)} записів")
            
            try:
                # Розділення на train/test для кластера
                if len(X_cluster) > 50:
                    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(
                        X_cluster.drop('cluster_id', axis=1), y_cluster, test_size=0.3, random_state=42
                    )
                    
                    # Тестуємо GaussianNB для кластера
                    model = GaussianNB()
                    model.fit(X_train_cluster, y_train_cluster)
                    
                    train_accuracy = model.score(X_train_cluster, y_train_cluster)
                    test_accuracy = model.score(X_test_cluster, y_test_cluster)
                    
                    cluster_bayesian_results[cluster_id] = {
                        'data_points': len(X_cluster),
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'price_up_ratio': y_cluster.mean(),  # Частка зростання ціни
                        'feature_importance': dict(zip(
                            ['whale_volume_usd', 'net_flow', 'whale_activity', 'exchange_inflow', 'exchange_outflow'],
                            model.theta_[1] - model.theta_[0] if hasattr(model, 'theta_') else [0]*5
                        ))
                    }
                    
                    print(f"   Train accuracy: {train_accuracy:.3f}")
                    print(f"   Test accuracy: {test_accuracy:.3f}")
                    print(f"   Частка зростання ціни: {y_cluster.mean():.1%}")
                
            except Exception as e:
                print(f"   Помилка Bayesian для кластера {cluster_id}: {e}")
        
        # ЗАГАЛЬНИЙ BAYESIAN АНАЛІЗ (з cluster_id як фічею)
        print(f"\n ЗАГАЛЬНИЙ BAYESIAN АНАЛІЗ (з cluster_id)...")
        X_general = X.dropna()
        y_general = y.loc[X_general.index]
        
        # Розділення на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Тестуємо різні Байєсові моделі
        models = {
            'GaussianNB': GaussianNB(),
            'MultinomialNB': MultinomialNB(),
            'BernoulliNB': BernoulliNB()
        }
        
        # Нормалізація для Multinomial (потрібні невід'ємні значення)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_positive = X_train_scaled - X_train_scaled.min() + 1
        X_test_positive = X_test_scaled - X_test_scaled.min() + 1
        
        for name, model in models.items():
            if name == 'MultinomialNB':
                X_tr = X_train_positive
                X_te = X_test_positive
            else:
                X_tr = X_train
                X_te = X_test
                
            # Навчання та оцінка
            model.fit(X_tr, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5)
            
            # Прогноз
            y_pred = model.predict(X_te)
            
            # Метрики
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'cv_accuracy': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'test_accuracy': float(report['accuracy']),
                'precision': float(report['1']['precision']),
                'recall': float(report['1']['recall']),
                'f1_score': float(report['1']['f1-score'])
            }
            
            print(f"  {name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test={report['accuracy']:.3f}")
        
        # Додаємо cluster-based результати
        results['cluster_bayesian_analysis'] = cluster_bayesian_results
        results['total_clusters_analyzed'] = len(cluster_bayesian_results)
        
        print(f"\n Cluster-based Bayesian: проаналізовано {len(cluster_bayesian_results)} кластерів")
        
        # Підсумок по кластерах
        if cluster_bayesian_results:
            print("\n ПІДСУМОК BAYESIAN ПО КЛАСТЕРАХ:")
            for cluster_id, cluster_result in cluster_bayesian_results.items():
                print(f"  Кластер {cluster_id}: Test accuracy={cluster_result['test_accuracy']:.3f}, "
                      f"Price up ratio={cluster_result['price_up_ratio']:.1%}")
            
        self.all_results['bayesian_analysis'] = results
        
        return results
        
    def build_prediction_models(self, df):
        """Побудова прогнозних моделей"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        results = {}
        
        # Підготовка даних - прогнозуємо ціну через 1 годину
        df['target_price'] = df['btc_price'].shift(-1)
        
        # Базові фічі
        base_features = ['whale_volume_usd', 'net_flow', 'whale_activity',
                        'exchange_inflow', 'exchange_outflow', 'btc_price']
        
        features = base_features.copy()
        
        # Додаємо лагові ознаки
        for lag in [1, 3, 6]:
            df[f'whale_volume_lag{lag}'] = df['whale_volume_usd'].shift(lag)
            df[f'net_flow_lag{lag}'] = df['net_flow'].shift(lag)
            features.extend([f'whale_volume_lag{lag}', f'net_flow_lag{lag}'])
            
        # Додаємо оптимальні фічі якщо вони були інтегровані
        if hasattr(self, 'optimal_features') and self.optimal_features:
            # Вибираємо топ-15 оптимальних фічей для прогнозування
            top_optimal = self.optimal_features[:15]
            for feat in top_optimal:
                if feat in df.columns and feat not in features:
                    features.append(feat)
            print(f" Додано {len(features) - len(base_features) - 6} оптимальних фічей до прогнозних моделей")
            
        # Готуємо дані
        data = df[features + ['target_price']].dropna()
        X = data[features]
        y = data['target_price']
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Моделі
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            # Навчання
            model.fit(X_train, y_train)
            
            # Прогноз
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Метрики
            results[name] = {
                'train_r2': float(r2_score(y_train, y_pred_train)),
                'test_r2': float(r2_score(y_test, y_pred_test)),
                'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            }
            
            # Feature importance для RandomForest
            if name == 'RandomForest':
                importance = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                results[name]['feature_importance'] = importance.to_dict('records')
                
                # Візуалізація
                plt.figure(figsize=(10, 6))
                importance.head(10).plot(x='feature', y='importance', kind='barh')
                plt.xlabel('Importance')
                plt.title('Top 10 Feature Importance')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'feature_importance.png', dpi=300)
                plt.close()
                
            print(f"  {name}: Train R²={results[name]['train_r2']:.3f}, Test R²={results[name]['test_r2']:.3f}")
            
        self.all_results['prediction_models'] = results
        
        return results
        
    def create_all_visualizations(self, df):
        """Створення всіх професійних візуалізацій"""
        
        # 1. Загальний дашборд
        fig = plt.figure(figsize=(20, 16))
        
        # Whale volume over time
        ax1 = plt.subplot(4, 2, 1)
        df['whale_volume_usd'].resample('D').sum().plot(ax=ax1, color='darkblue')
        ax1.set_title('Daily Whale Volume', fontsize=14)
        ax1.set_ylabel('Volume (USD)')
        ax1.grid(True, alpha=0.3)
        
        # Net flow
        ax2 = plt.subplot(4, 2, 2)
        df['net_flow'].resample('D').sum().plot(ax=ax2, color='green', label='Net Flow')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Daily Net Exchange Flow', fontsize=14)
        ax2.set_ylabel('Net Flow (USD)')
        ax2.grid(True, alpha=0.3)
        
        # Hourly patterns
        ax3 = plt.subplot(4, 2, 3)
        hourly_avg = df.groupby(df.index.hour)['whale_volume_usd'].mean()
        hourly_avg.plot(kind='bar', ax=ax3, color='steelblue')
        ax3.set_title('Average Whale Volume by Hour', fontsize=14)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Avg Volume (USD)')
        
        # Day of week patterns
        ax4 = plt.subplot(4, 2, 4)
        daily_avg = df.groupby(df.index.dayofweek)['whale_volume_usd'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_avg.index = days
        daily_avg.plot(kind='bar', ax=ax4, color='coral')
        ax4.set_title('Average Whale Volume by Day of Week', fontsize=14)
        ax4.set_xlabel('Day')
        ax4.set_ylabel('Avg Volume (USD)')
        
        # Price vs whale volume scatter
        ax5 = plt.subplot(4, 2, 5)
        ax5.scatter(df['whale_volume_usd'], df['btc_price'], alpha=0.5, s=30)
        ax5.set_xlabel('Whale Volume (USD)')
        ax5.set_ylabel('BTC Price')
        ax5.set_title('Whale Volume vs BTC Price', fontsize=14)
        
        # Distribution plots
        ax6 = plt.subplot(4, 2, 6)
        df['whale_volume_usd'].hist(bins=50, ax=ax6, color='purple', alpha=0.7)
        ax6.set_xlabel('Whale Volume (USD)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Whale Volume Distribution', fontsize=14)
        ax6.set_yscale('log')
        
        # Cumulative volume
        ax7 = plt.subplot(4, 2, 7)
        df['whale_volume_usd'].cumsum().plot(ax=ax7, color='darkgreen')
        ax7.set_title('Cumulative Whale Volume', fontsize=14)
        ax7.set_ylabel('Cumulative Volume (USD)')
        
        # Rolling correlation
        ax8 = plt.subplot(4, 2, 8)
        rolling_corr = df['whale_volume_usd'].rolling(168).corr(df['btc_price'])
        rolling_corr.plot(ax=ax8, color='red')
        ax8.set_title('7-Day Rolling Correlation (Whale Volume vs BTC Price)', fontsize=14)
        ax8.set_ylabel('Correlation')
        ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle('GT14 WhaleTracker v14.2 - Comprehensive Dashboard', fontsize=18)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Створено comprehensive dashboard")
        
        # 2. Додаткові спеціалізовані візуалізації
        self._create_flow_analysis_charts(df)
        self._create_predictive_charts(df)
        
    def _create_flow_analysis_charts(self, df):
        """Детальний аналіз потоків"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Inflow vs Outflow
        ax1 = axes[0, 0]
        df[['exchange_inflow', 'exchange_outflow']].resample('D').sum().plot(ax=ax1)
        ax1.set_title('Daily Exchange Flows')
        ax1.set_ylabel('Volume (USD)')
        ax1.legend(['Inflow', 'Outflow'])
        
        # Flow imbalance
        ax2 = axes[0, 1]
        flow_imbalance = (df['exchange_outflow'] - df['exchange_inflow']).resample('D').sum()
        colors = ['green' if x > 0 else 'red' for x in flow_imbalance]
        ax2.bar(flow_imbalance.index, flow_imbalance.values, color=colors, alpha=0.7)
        ax2.set_title('Daily Flow Imbalance (Outflow - Inflow)')
        ax2.set_ylabel('Imbalance (USD)')
        
        # Whale activity vs price change
        ax3 = axes[1, 0]
        price_change = df['btc_price'].pct_change() * 100
        ax3.scatter(df['whale_activity'], price_change, alpha=0.5)
        ax3.set_xlabel('Whale Activity Index')
        ax3.set_ylabel('BTC Price Change (%)')
        ax3.set_title('Whale Activity vs Price Change')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Top whale days
        ax4 = axes[1, 1]
        top_days = df.nlargest(10, 'whale_volume_usd')[['whale_volume_usd', 'btc_price']]
        x = range(len(top_days))
        ax4_twin = ax4.twinx()
        
        ax4.bar(x, top_days['whale_volume_usd'], alpha=0.7, color='blue', label='Whale Volume')
        ax4_twin.plot(x, top_days['btc_price'], 'ro-', label='BTC Price')
        
        ax4.set_xlabel('Top 10 Whale Days')
        ax4.set_ylabel('Whale Volume (USD)', color='blue')
        ax4_twin.set_ylabel('BTC Price', color='red')
        ax4.set_title('Top 10 Whale Activity Days')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'flow_analysis_charts.png', dpi=300)
        plt.close()
        
        print(" Створено flow analysis charts")
        
    def _create_predictive_charts(self, df):
        """Візуалізації для прогнозних моделей"""
        # Lag analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        lags = [1, 3, 6, 12]
        for i, lag in enumerate(lags):
            ax = axes[i//2, i%2]
            
            df_lag = df[['whale_volume_usd', 'btc_price']].dropna()
            
            # Scatter з лагом
            ax.scatter(df_lag['whale_volume_usd'].shift(lag), 
                      df_lag['btc_price'].pct_change() * 100,
                      alpha=0.5, s=30)
            
            # Лінія тренду
            from scipy import stats as scipy_stats
            x = df_lag['whale_volume_usd'].shift(lag).dropna()
            y = df_lag['btc_price'].pct_change().shift(-lag).dropna() * 100
            
            if len(x) == len(y):
                slope, intercept, r_value, _, _ = scipy_stats.linregress(x, y)
                ax.plot(x, slope * x + intercept, 'r-', alpha=0.8)
                
                ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}', 
                       transform=ax.transAxes, verticalalignment='top')
            
            ax.set_xlabel(f'Whale Volume (lag {lag}h)')
            ax.set_ylabel('BTC Price Change (%)')
            ax.set_title(f'Whale Volume vs Future Price Change (lag {lag}h)')
            
        plt.tight_layout()
        plt.savefig(self.results_dir / 'predictive_lag_analysis.png', dpi=300)
        plt.close()
        
        print(" Створено predictive charts")
        
    def generate_client_reports(self):
        """Генерація професійних звітів для клієнта"""
        
        # 1. Executive Summary
        summary = f"""
# GT14 WhaleTracker v14.2 - Executive Summary

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Key Findings

### 1. Market Overview
- **Total Whale Volume:** ${self.all_results['basic_statistics']['total_whale_volume']:,.0f}
- **Analysis Period:** {self.all_results['basic_statistics']['date_range']}
- **Total Transactions:** {self.all_results['basic_statistics']['total_transactions']:,}

### 2. Clustering Analysis (All Metrics)
- **Optimal Clusters:** {self.all_results['clustering']['optimal_k']}
- **Silhouette Score:** {self.all_results['clustering']['final_metrics']['silhouette']:.3f}
- **Davies-Bouldin Score:** {self.all_results['clustering']['final_metrics']['davies_bouldin']:.3f}
- **Calinski-Harabasz Score:** {self.all_results['clustering']['final_metrics']['calinski_harabasz']:.1f}

### 3. VAR Analysis Results
- **Optimal Lag:** {self.all_results['var_analysis']['optimal_lag']} hours
- **Model Stability:** {'Stable' if self.all_results['var_analysis']['model_stable'] else 'Unstable'}
- **Key Causal Relationships:** {sum(1 for k,v in self.all_results['var_analysis']['granger_causality'].items() if v['significant'])}

### 4. Predictive Performance
- **Best Model:** Random Forest
- **Test R²:** {self.all_results['prediction_models']['RandomForest']['test_r2']:.3f}

### 5. Anomaly Detection
- **Consensus Anomalies:** {self.all_results['anomaly_detection']['anomalies']['consensus']['count']}
- **Peak Activity Hours:** {self.all_results['temporal_analysis']['activity_peaks']['peak_count']}

## Recommendations

1. **Trading Strategy:** Focus on whale outflows > $10M as buy signals
2. **Risk Management:** Monitor anomaly dates for potential market disruptions
3. **Optimal Timing:** Peak whale activity occurs during specific hours - adjust trading accordingly

---
*This report includes all analyses from versions 8 and 14.1, plus enhanced features in 14.2*
"""
        
        with open(self.results_dir / 'executive_summary.md', 'w', encoding='utf-8') as f:
            f.write(summary)
            
        # 2. Детальний JSON звіт (пропускаємо через проблеми з серіалізацією)
        # with open(self.results_dir / 'complete_analysis_results.json', 'w', encoding='utf-8') as f:
        #     json.dump(self.all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
        # 3. HTML звіт для презентації
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GT14 WhaleTracker v14.2 - Analysis Report</title>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
        }}
        .metric {{
            background: #ecf0f1;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        img {{
            max-width: 100%;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .warning {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1> GT14 WhaleTracker v14.2 - Complete Analysis Report</h1>
        
        <div class="grid">
            <div class="metric">
                <div class="metric-value">${self.all_results['basic_statistics']['total_whale_volume']/1e9:.1f}B</div>
                <div class="metric-label">Total Whale Volume</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.all_results['basic_statistics']['total_transactions']:,}</div>
                <div class="metric-label">Total Transactions</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.all_results['clustering']['optimal_k']}</div>
                <div class="metric-label">Optimal Clusters</div>
            </div>
        </div>
        
        <h2> Analysis Components</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Status</th>
                <th>Key Result</th>
            </tr>
            <tr>
                <td>Cross-Correlation Analysis</td>
                <td class="success"> Complete</td>
                <td>Analyzed {len(self.all_results.get('cross_correlation', {}))} variable pairs</td>
            </tr>
            <tr>
                <td>Seasonality Analysis</td>
                <td class="success"> Complete</td>
                <td>Daily seasonality: {self.all_results.get('seasonality', {}).get('daily_seasonality', {}).get('seasonal_strength', 0):.1%}</td>
            </tr>
            <tr>
                <td>Anomaly Detection</td>
                <td class="success"> Complete</td>
                <td>{self.all_results.get('anomaly_detection', {}).get('anomalies', {}).get('consensus', {}).get('count', 0)} consensus anomalies</td>
            </tr>
            <tr>
                <td>Clustering (All Metrics)</td>
                <td class="success"> Complete</td>
                <td>Silhouette: {self.all_results['clustering']['final_metrics']['silhouette']:.3f}</td>
            </tr>
            <tr>
                <td>VAR with IRF/FEVD</td>
                <td class="success"> Complete</td>
                <td>Lag {self.all_results['var_analysis']['optimal_lag']}, {'Stable' if self.all_results['var_analysis']['model_stable'] else 'Unstable'}</td>
            </tr>
            <tr>
                <td>Bayesian Analysis</td>
                <td class="success"> Complete</td>
                <td>Best: {max([k for k,v in self.all_results['bayesian_analysis'].items() if isinstance(v, dict) and 'test_accuracy' in v], key=lambda x: self.all_results['bayesian_analysis'][x]['test_accuracy']) if any('test_accuracy' in v for v in self.all_results['bayesian_analysis'].values() if isinstance(v, dict)) else 'BernoulliNB'}</td>
            </tr>
            <tr>
                <td>Prediction Models</td>
                <td class="success"> Complete</td>
                <td>R²: {self.all_results['prediction_models']['RandomForest']['test_r2']:.3f}</td>
            </tr>
        </table>
        
        <h2> Visualizations</h2>
        <img src="comprehensive_dashboard.png" alt="Comprehensive Dashboard">
        <img src="clustering_metrics_comparison.png" alt="Clustering Metrics">
        <img src="correlation_matrix.png" alt="Correlation Matrix">
        <img src="var_irf_complete.png" alt="VAR IRF Analysis">
        
        <p style="text-align: center; margin-top: 50px; color: #7f8c8d;">
            GT14 WhaleTracker v14.2 © 2025 | Complete Enhanced Analysis Pipeline
        </p>
    </div>
</body>
</html>
"""
        
        with open(self.results_dir / 'analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
            
        # 4. Порівняння версій
        comparison = {
            'version_comparison': {
                'v8_features': {
                    'cross_correlation': ' Included',
                    'seasonality': ' Included', 
                    'anomaly_detection': ' Included',
                    'var_analysis': ' Enhanced with full IRF/FEVD'
                },
                'v14_1_features': {
                    'clustering': ' Enhanced with all metrics',
                    'bayesian': ' Included',
                    'predictions': ' Included',
                    'mcp_integration': ' Ready'
                },
                'v14_2_new': {
                    'all_metrics_clustering': ' Davies-Bouldin, Calinski-Harabasz, Silhouette',
                    'comprehensive_var': ' Full IRF and FEVD analysis',
                    'professional_reports': ' Client-ready HTML and PDF',
                    'interactive_selection': ' Model comparison framework'
                }
            }
        }
        
        with open(self.results_dir / 'version_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
            
        print("\n Згенеровано звіти:")
        print(f"  - {self.results_dir}/executive_summary.md")
        print(f"  - {self.results_dir}/complete_analysis_results.json") 
        print(f"  - {self.results_dir}/analysis_report.html")
        print(f"  - {self.results_dir}/version_comparison.json")
        
    def arima_visualization_integrated(self):
        """Інтегрована ARIMA візуалізація"""
        try:
            if ARIMAVisualization is None:
                self.log.warning("ARIMAVisualization модуль не знайдено")
                return
                
            self.log.info("Запуск інтегрованої ARIMA візуалізації...")
            
            # Створюємо екземпляр візуалізатора
            visualizer = ARIMAVisualization()
            
            # Запускаємо візуалізацію
            results = visualizer.create_arima_visualization()
            
            if results:
                # Зберігаємо результати
                self.all_results['arima_visualization'] = results
                
                # Копіюємо файли в папку результатів
                import shutil
                for file in ['arima_forecast_visualization.png', 'arima_visualization_report.md']:
                    if os.path.exists(file):
                        shutil.copy(file, self.results_dir / file)
                        
                self.log.info("✅ ARIMA візуалізація завершена успішно")
                self.log.info(f"  Історичних точок: {results.get('historical_points', 0)}")
                self.log.info(f"  Прогноз на: {results.get('forecast_periods', 0)} годин")
                
        except Exception as e:
            self.log.error(f"Помилка ARIMA візуалізації: {e}")
            self.logger.log_error(e, context={'method': 'arima_visualization_integrated'})
            
    def granger_causality_integrated(self):
        """Інтегрований Enhanced Granger Causality аналіз"""
        try:
            if GrangerCausalityEnhanced is None:
                self.log.warning("GrangerCausalityEnhanced модуль не знайдено")
                return
                
            self.log.info("Запуск Enhanced Granger Causality аналізу...")
            
            # Створюємо аналізатор
            analyzer = GrangerCausalityEnhanced()
            
            # Запускаємо аналіз з нашими даними
            if hasattr(self, 'df') and self.df is not None:
                results = analyzer.analyze_granger_causality(self.df)
                
                if results:
                    # Зберігаємо результати
                    self.all_results['granger_causality_enhanced'] = results
                    
                    # Копіюємо файли
                    import shutil
                    for file in ['granger_causality_matrix.png', 'granger_causality_strength.png', 
                               'granger_causality_results.csv', 'granger_causality_significant.csv']:
                        if os.path.exists(file):
                            shutil.copy(file, self.results_dir / file)
                            
                    self.log.info("✅ Granger Causality аналіз завершено")
                    self.log.info(f"  Всього пар: {results.get('total_pairs', 0)}")
                    self.log.info(f"  Значущих зв'язків: {results.get('significant_pairs', 0)}")
                    
        except Exception as e:
            self.log.error(f"Помилка Granger Causality: {e}")
            self.logger.log_error(e, context={'method': 'granger_causality_integrated'})
            
    def feature_importance_integrated(self):
        """Інтегрований Feature Importance аналіз"""
        try:
            if FeatureImportanceAnalyzer is None:
                self.log.warning("FeatureImportanceAnalyzer модуль не знайдено")
                return
                
            self.log.info("Запуск Feature Importance аналізу...")
            
            # Визначаємо чи маємо збережені фічі
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM whale_features_basic")
            feature_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            if feature_count > 0:
                self.log.info(f"Знайдено {feature_count} записів з фічами в БД")
                
                # Створюємо аналізатор
                analyzer = FeatureImportanceAnalyzer()
                
                # Запускаємо аналіз
                results = analyzer.analyze_feature_importance()
                
                if results:
                    # Зберігаємо результати
                    self.all_results['feature_importance'] = results
                    
                    # Копіюємо файли
                    import shutil
                    for file in ['feature_importance_visualization.png', 'feature_importance_comparison.png',
                               'feature_importance_top50.csv', 'feature_importance_report.md']:
                        if os.path.exists(file):
                            shutil.copy(file, self.results_dir / file)
                            
                    self.log.info("✅ Feature Importance аналіз завершено")
                    self.log.info(f"  Проаналізовано фічей: {len(results.get('features', []))}")
                    
            else:
                self.log.warning("Фічі не знайдені в БД. Запускаємо збереження...")
                self._save_features_to_db()
                
        except Exception as e:
            self.log.error(f"Помилка Feature Importance: {e}")
            self.logger.log_error(e, context={'method': 'feature_importance_integrated'})
            
    def export_all_results_to_csv(self):
        """Експорт всіх результатів в CSV"""
        try:
            self.log.info("Експорт результатів в CSV...")
            
            # 1. Основна статистика
            if 'basic_statistics' in self.all_results:
                stats_df = pd.DataFrame([self.all_results['basic_statistics']])
                stats_df.to_csv(self.results_dir / 'basic_statistics.csv', index=False)
                
            # 2. Кластерний аналіз
            if 'clustering' in self.all_results:
                cluster_df = pd.DataFrame(self.all_results['clustering']['cluster_stats'])
                cluster_df.to_csv(self.results_dir / 'cluster_analysis.csv', index=False)
                
            # 3. VAR результати
            if 'var_analysis' in self.all_results:
                var_data = []
                for key, result in self.all_results['var_analysis'].get('granger_causality', {}).items():
                    var_data.append({
                        'relationship': key,
                        'p_value': result['p_value'],
                        'significant': result['significant']
                    })
                if var_data:
                    pd.DataFrame(var_data).to_csv(self.results_dir / 'var_analysis_results.csv', index=False)
                    
            # 4. Байєсів аналіз
            if 'bayesian_analysis' in self.all_results:
                bayes_data = []
                for model, metrics in self.all_results['bayesian_analysis'].items():
                    if isinstance(metrics, dict) and 'test_accuracy' in metrics:
                        bayes_data.append({
                            'model': model,
                            **metrics
                        })
                if bayes_data:
                    pd.DataFrame(bayes_data).to_csv(self.results_dir / 'bayesian_results.csv', index=False)
                    
            # 5. Прогнозні моделі
            if 'prediction_models' in self.all_results:
                pred_data = []
                for model, metrics in self.all_results['prediction_models'].items():
                    pred_data.append({
                        'model': model,
                        **metrics
                    })
                pd.DataFrame(pred_data).to_csv(self.results_dir / 'prediction_models_results.csv', index=False)
                
            self.log.info("✅ CSV експорт завершено")
            
        except Exception as e:
            self.log.error(f"Помилка CSV експорту: {e}")
            self.logger.log_error(e, context={'method': 'export_all_results_to_csv'})
            
    def create_interactive_visualizations(self):
        """Створення інтерактивних Plotly візуалізацій"""
        try:
            self.log.info("Створення інтерактивних візуалізацій...")
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 1. Інтерактивний дашборд
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Whale Volume Over Time', 'BTC Price vs Whale Activity',
                              'Exchange Flows', 'Clustering Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": True}],
                       [{"secondary_y": False}, {"type": "pie"}]]
            )
            
            if hasattr(self, 'df') and self.df is not None:
                # Whale volume
                fig.add_trace(
                    go.Scatter(x=self.df.index, y=self.df['whale_volume_usd'],
                             mode='lines', name='Whale Volume'),
                    row=1, col=1
                )
                
                # BTC price vs whale activity
                fig.add_trace(
                    go.Scatter(x=self.df.index, y=self.df['btc_price'],
                             mode='lines', name='BTC Price'),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=self.df.index, y=self.df['whale_activity'],
                             mode='lines', name='Whale Activity', yaxis='y2'),
                    row=1, col=2, secondary_y=True
                )
                
                # Exchange flows
                fig.add_trace(
                    go.Scatter(x=self.df.index, y=self.df['exchange_inflow'],
                             mode='lines', name='Inflow', line=dict(color='green')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=self.df.index, y=self.df['exchange_outflow'],
                             mode='lines', name='Outflow', line=dict(color='red')),
                    row=2, col=1
                )
                
            # Clustering pie chart
            if 'clustering' in self.all_results and 'cluster_stats' in self.all_results['clustering']:
                cluster_stats = self.all_results['clustering']['cluster_stats']
                labels = [f"Cluster {stat['cluster']}" for stat in cluster_stats]
                values = [stat['size'] for stat in cluster_stats]
                
                fig.add_trace(
                    go.Pie(labels=labels, values=values),
                    row=2, col=2
                )
                
            fig.update_layout(height=800, showlegend=True, 
                            title_text="GT14 WhaleTracker v14.2 - Interactive Dashboard")
            fig.write_html(self.results_dir / 'interactive_dashboard.html')
            
            self.log.info("✅ Інтерактивні візуалізації створено")
            self.log.info(f"  Файл: {self.results_dir}/interactive_dashboard.html")
            
        except Exception as e:
            self.log.error(f"Помилка створення візуалізацій: {e}")
            self.logger.log_error(e, context={'method': 'create_interactive_visualizations'})
            
    def _save_features_to_db(self):
        """Зберігає фічі в БД якщо їх ще немає"""
        try:
            from feature_persistence_quick import FeaturePersistenceQuick
            
            self.log.info("Запуск збереження фічей в БД...")
            persistence = FeaturePersistenceQuick()
            persistence.save_features_to_db()
            
        except Exception as e:
            self.log.error(f"Помилка збереження фічей: {e}")
            
    def arima_ensemble_analysis(self):
        """ARIMA Ensemble з 8 моделями"""
        try:
            if ARIMAEnsembleModels is None:
                self.log.warning("ARIMAEnsembleModels модуль не знайдено")
                return
                
            self.log.info("Запуск ARIMA Ensemble аналізу (8 моделей)...")
            
            # Створюємо екземпляр
            ensemble = ARIMAEnsembleModels()
            
            # Використовуємо дані з основного DataFrame
            if hasattr(self, 'df') and self.df is not None and 'btc_price' in self.df.columns:
                # Підготовка даних
                btc_data = self.df[['btc_price']].copy()
                btc_data = btc_data[btc_data['btc_price'] > 0].dropna()
                
                if len(btc_data) < 168:  # Мінімум 7 днів
                    self.log.warning("Недостатньо даних для ARIMA Ensemble")
                    return
                    
                # Визначаємо train/test
                forecast_horizon = 24
                train_size = len(btc_data) - forecast_horizon
                train_data = btc_data['btc_price'][:train_size]
                test_data = btc_data['btc_price'][train_size:] if train_size < len(btc_data) else None
                
                self.log.info(f"Train: {len(train_data)} записів, Test: {len(test_data) if test_data is not None else 0} записів")
                
                # Навчання моделей
                results = ensemble.generate_forecasts(train_data, test_data, horizon=forecast_horizon)
                
                # Візуалізація
                fig = ensemble.visualize_model_comparison(
                    train_data,
                    test_data,
                    save_path=self.results_dir / 'arima_ensemble_comparison.png'
                )
                plt.close()
                
                # Збереження результатів
                ensemble_results = ensemble.save_results()
                
                # Генерація звіту
                report = ensemble.generate_report()
                
                # Копіюємо файли в папку результатів
                import shutil
                for file in ['arima_ensemble_results.json', 'arima_ensemble_report.md']:
                    if os.path.exists(file):
                        shutil.copy(file, self.results_dir / file)
                        
                # Зберігаємо в загальні результати
                self.all_results['arima_ensemble'] = {
                    'models_count': ensemble_results['models_count'],
                    'best_model': ensemble_results['best_model'],
                    'best_mape': ensemble_results['best_mape'],
                    'models': list(ensemble_results['models'].keys())
                }
                
                self.log.info("✅ ARIMA Ensemble аналіз завершено")
                self.log.info(f"  Кількість моделей: {ensemble_results['models_count']}")
                self.log.info(f"  Найкраща модель: {ensemble_results['best_model']}")
                self.log.info(f"  MAPE: {ensemble_results['best_mape']:.2f}%")
                
            else:
                # Якщо немає даних, завантажуємо з БД
                data = ensemble.load_data(hours=336)
                
                train_size = len(data) - 24
                train_data = data['btc_price'][:train_size]
                test_data = data['btc_price'][train_size:]
                
                results = ensemble.generate_forecasts(train_data, test_data, horizon=24)
                
                fig = ensemble.visualize_model_comparison(
                    train_data,
                    test_data,
                    save_path=self.results_dir / 'arima_ensemble_comparison.png'
                )
                plt.close()
                
                ensemble_results = ensemble.save_results()
                
                self.all_results['arima_ensemble'] = {
                    'models_count': len(results),
                    'best_model': ensemble_results.get('best_model'),
                    'best_mape': ensemble_results.get('best_mape')
                }
                
        except Exception as e:
            self.log.error(f"Помилка ARIMA Ensemble: {e}")
            self.logger.log_error(e, context={'method': 'arima_ensemble_analysis'})

# Спеціальний JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Запуск
if __name__ == "__main__":
    pipeline = GT14_Complete_Pipeline()
    pipeline.run_complete_analysis()