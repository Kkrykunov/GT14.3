#!/usr/bin/env python3
"""
Комплексні тести для GT14 v14.3
Покриття всіх основних модулів та функціональності
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mysql.connector
from unittest.mock import patch, MagicMock
import json
import tempfile
import shutil


class TestDatabaseConnection(unittest.TestCase):
    """Тести підключення до БД"""
    
    def setUp(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        
    def test_connection(self):
        """Тест з'єднання з БД"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            self.assertTrue(conn.is_connected())
            conn.close()
        except Exception as e:
            self.fail(f"Failed to connect to database: {e}")
            
    def test_tables_exist(self):
        """Перевірка наявності основних таблиць"""
        required_tables = [
            'whale_hourly_complete',
            'whale_alerts_original',
            'universal_features',
            'whale_features_basic',
            'cluster_labels',
            'arima_models'
        ]
        
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        cursor.execute("SHOW TABLES")
        existing_tables = [table[0] for table in cursor.fetchall()]
        
        for table in required_tables:
            self.assertIn(table, existing_tables, f"Table {table} not found")
            
        cursor.close()
        conn.close()


class TestFeatureIntegration(unittest.TestCase):
    """Тести інтеграції фічей"""
    
    def test_feature_loading(self):
        """Тест завантаження фічей з БД"""
        from GT14_v14_2_COMPLETE_ENHANCED_PIPELINE import GT14_Complete_Pipeline
        
        pipeline = GT14_Complete_Pipeline()
        
        # Тест методу load_all_features_from_db
        try:
            df_features = pipeline.load_all_features_from_db()
            self.assertIsInstance(df_features, pd.DataFrame)
            self.assertGreater(df_features.shape[1], 200)  # 233+ фічей
            self.assertGreater(len(df_features), 0)
        except Exception as e:
            self.fail(f"Failed to load features: {e}")
            
    def test_optimal_features_selection(self):
        """Тест вибору оптимальних фічей"""
        # Створюємо тестові дані
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # Генеруємо фічі з різною кореляцією до цільової змінної
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        
        # Цільова змінна залежить від перших 10 фічей
        y = X.iloc[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.1
        
        # Додаємо btc_price
        df_features = X.copy()
        df_target = pd.DataFrame({'btc_price': y})
        
        from GT14_v14_2_COMPLETE_ENHANCED_PIPELINE import GT14_Complete_Pipeline
        pipeline = GT14_Complete_Pipeline()
        
        # Аналіз фічей
        df_analysis = pipeline.analyze_individual_features(df_features, df_target)
        
        # Перевірка
        self.assertIsInstance(df_analysis, pd.DataFrame)
        self.assertEqual(len(df_analysis), n_features)
        
        # Топ фічі повинні бути з перших 10
        top_10 = df_analysis.head(10).index.tolist()
        original_top_features = [f'feature_{i}' for i in range(10)]
        
        # Принаймні 5 з топ-10 повинні бути з оригінальних
        intersection = set(top_10) & set(original_top_features)
        self.assertGreaterEqual(len(intersection), 5)


class TestARIMAEnsemble(unittest.TestCase):
    """Тести ARIMA Ensemble моделей"""
    
    def setUp(self):
        # Створюємо тестові дані
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
        
        # Синусоїдальний тренд + шум
        trend = np.sin(np.arange(500) * 0.1) * 1000 + 50000
        noise = np.random.randn(500) * 100
        self.test_data = pd.Series(trend + noise, index=dates, name='btc_price')
        
    def test_ensemble_initialization(self):
        """Тест ініціалізації ensemble"""
        from arima_ensemble_models import ARIMAEnsembleModels
        
        ensemble = ARIMAEnsembleModels()
        self.assertEqual(len(ensemble.models_config), 8)
        self.assertIn('ARIMA_111', ensemble.models_config)
        self.assertIn('SARIMA_111_111_24', ensemble.models_config)
        self.assertIn('HoltWinters_add', ensemble.models_config)
        
    def test_model_fitting(self):
        """Тест навчання моделей"""
        from arima_ensemble_models import ARIMAEnsembleModels
        
        ensemble = ARIMAEnsembleModels()
        
        # Навчання на тестових даних
        train_data = self.test_data[:400]
        test_data = self.test_data[400:]
        
        results = ensemble.generate_forecasts(train_data, test_data, horizon=24)
        
        # Перевірка результатів
        self.assertGreater(len(results), 0)
        
        for model_name, result in results.items():
            self.assertIn('forecast', result)
            self.assertIn('metrics', result)
            self.assertEqual(len(result['forecast']), 24)
            
    def test_ensemble_forecast(self):
        """Тест ансамблевого прогнозу"""
        from arima_ensemble_models import ARIMAEnsembleModels
        
        ensemble = ARIMAEnsembleModels()
        
        # Підготовка даних
        train_data = self.test_data[:400]
        test_data = self.test_data[400:]
        
        # Навчання
        ensemble.generate_forecasts(train_data, test_data, horizon=24)
        
        # Ансамблевий прогноз
        ensemble_forecast = ensemble.create_ensemble_forecast()
        
        self.assertEqual(len(ensemble_forecast), 24)
        self.assertIsInstance(ensemble_forecast, pd.Series)


class TestGrangerCausality(unittest.TestCase):
    """Тести Enhanced Granger Causality"""
    
    def setUp(self):
        # Створюємо тестові дані з причинними зв'язками
        np.random.seed(42)
        n = 1000
        
        # X впливає на Y з лагом 2
        self.X = np.random.randn(n)
        self.Y = np.zeros(n)
        
        for i in range(2, n):
            self.Y[i] = 0.7 * self.X[i-2] + 0.3 * self.Y[i-1] + np.random.randn() * 0.1
            
        self.df = pd.DataFrame({
            'X': self.X,
            'Y': self.Y,
            'btc_price': self.Y * 1000 + 50000,
            'whale_volume_usd': np.abs(self.X) * 1e6,
            'net_flow': self.X * 1e5
        })
        
    def test_granger_analysis(self):
        """Тест Granger causality аналізу"""
        from granger_causality_enhanced import GrangerCausalityEnhanced
        
        analyzer = GrangerCausalityEnhanced()
        
        # Аналіз з DataFrame
        results = analyzer.analyze_granger_causality(self.df, max_lag=5)
        
        self.assertIsNotNone(results)
        self.assertIn('total_pairs', results)
        self.assertIn('significant_pairs', results)
        self.assertGreater(results['significant_pairs'], 0)


class TestFeatureImportance(unittest.TestCase):
    """Тести Feature Importance аналізу"""
    
    @patch('mysql.connector.connect')
    def test_feature_importance_analysis(self, mock_connect):
        """Тест аналізу важливості фічей"""
        # Мокаємо БД з'єднання
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Мокаємо дані
        mock_cursor.fetchall.return_value = [
            ('feature_1', 100.0, 50000.0),
            ('feature_2', 200.0, 51000.0),
            ('feature_3', 150.0, 49000.0)
        ]
        
        from feature_importance_analysis import FeatureImportanceAnalyzer
        
        analyzer = FeatureImportanceAnalyzer()
        
        # Створюємо тестові дані для методів
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'btc_price': np.random.randn(100) * 1000 + 50000
        })
        
        # Тест RandomForest importance
        rf_importance = analyzer.calculate_rf_importance(test_data.drop('btc_price', axis=1), 
                                                        test_data['btc_price'])
        self.assertIsInstance(rf_importance, pd.DataFrame)
        self.assertEqual(len(rf_importance), 3)


class TestCompletePipeline(unittest.TestCase):
    """Тести повного pipeline"""
    
    def test_pipeline_initialization(self):
        """Тест ініціалізації pipeline"""
        from GT14_v14_2_COMPLETE_ENHANCED_PIPELINE import GT14_Complete_Pipeline
        
        pipeline = GT14_Complete_Pipeline()
        
        # Перевірка атрибутів
        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.results_dir)
        self.assertIsNotNone(pipeline.all_results)
        self.assertEqual(pipeline.all_results['metadata']['version'], '14.2')
        
    @patch('mysql.connector.connect')
    def test_pipeline_stages(self, mock_connect):
        """Тест послідовності етапів"""
        # Мокаємо БД
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.is_connected.return_value = True
        
        from GT14_v14_2_COMPLETE_ENHANCED_PIPELINE import GT14_Complete_Pipeline
        
        pipeline = GT14_Complete_Pipeline()
        
        # Перевірка наявності всіх методів
        required_methods = [
            'load_and_analyze_data',
            'temporal_analysis',
            'cross_correlation_analysis',
            'seasonality_analysis',
            'anomaly_detection',
            'advanced_clustering',
            'comprehensive_var_analysis',
            'bayesian_analysis',
            'build_prediction_models',
            'create_all_visualizations',
            'generate_client_reports',
            'arima_visualization_integrated',
            'granger_causality_integrated',
            'feature_importance_integrated',
            'export_all_results_to_csv',
            'create_interactive_visualizations',
            'arima_ensemble_analysis'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(pipeline, method), f"Method {method} not found")


class TestDataQuality(unittest.TestCase):
    """Тести якості даних"""
    
    def test_data_validation(self):
        """Тест валідації даних"""
        # Створюємо дані з проблемами
        df = pd.DataFrame({
            'btc_price': [50000, 51000, -100, 52000, np.nan, 53000],
            'whale_volume_usd': [1e6, 2e6, 3e6, np.nan, 5e6, -1e6],
            'timestamp': pd.date_range('2024-01-01', periods=6, freq='H')
        })
        
        # Очистка
        df_clean = df[df['btc_price'] > 0].dropna()
        
        # Перевірка
        self.assertEqual(len(df_clean), 3)
        self.assertTrue((df_clean['btc_price'] > 0).all())
        self.assertFalse(df_clean.isnull().any().any())


class TestIntegration(unittest.TestCase):
    """Інтеграційні тести"""
    
    def test_module_imports(self):
        """Тест імпорту всіх модулів"""
        modules = [
            'GT14_v14_2_COMPLETE_ENHANCED_PIPELINE',
            'arima_visualization',
            'granger_causality_enhanced',
            'feature_importance_analysis',
            'arima_ensemble_models',
            'universal_feature_engineering',
            'self_learning_arima',
            'cluster_detailed_analysis',
            'feature_persistence_quick'
        ]
        
        for module in modules:
            try:
                __import__(module)
            except ImportError as e:
                self.fail(f"Failed to import {module}: {e}")
                
    def test_results_directory_creation(self):
        """Тест створення директорії результатів"""
        from GT14_v14_2_COMPLETE_ENHANCED_PIPELINE import GT14_Complete_Pipeline
        
        pipeline = GT14_Complete_Pipeline()
        
        # Перевірка створення директорії
        self.assertTrue(pipeline.results_dir.exists())
        self.assertTrue(pipeline.results_dir.is_dir())


def run_all_tests():
    """Запуск всіх тестів"""
    # Створюємо test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Додаємо всі тест-класи
    test_classes = [
        TestDatabaseConnection,
        TestFeatureIntegration,
        TestARIMAEnsemble,
        TestGrangerCausality,
        TestFeatureImportance,
        TestCompletePipeline,
        TestDataQuality,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    # Запускаємо тести
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Підсумок
    print("\n" + "="*60)
    print("ПІДСУМОК ТЕСТУВАННЯ")
    print("="*60)
    print(f"Всього тестів: {result.testsRun}")
    print(f"Успішно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Помилок: {len(result.failures)}")
    print(f"Критичних помилок: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)