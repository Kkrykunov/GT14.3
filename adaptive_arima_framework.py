#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive ARIMA Framework для GT14 v14.2
Самонавчальна система з автоматичним вибором параметрів
"""

import pandas as pd
import numpy as np
import mysql.connector
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf, acf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


class AdaptiveARIMAFramework:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        self.models = {}
        self.performance_history = {}
        self.best_params = {}
        self.ensemble_weights = {}
        self.retrain_interval = 24  # години
        self.window_size = 168  # 7 днів для навчання
        
    def load_data(self):
        """Завантаження даних з БД"""
        print("=== Завантаження даних ===")
        
        conn = mysql.connector.connect(**self.db_config)
        
        # Завантажуємо основні дані
        query = """
        SELECT timestamp, btc_price, whale_volume_usd, net_flow, 
               whale_activity, fear_greed_index
        FROM whale_hourly_complete
        ORDER BY timestamp
        """
        
        self.data = pd.read_sql(query, conn)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data.set_index('timestamp', inplace=True)
        
        # Завантажуємо найкращі фічі з попередніх етапів
        try:
            query = """
            SELECT DISTINCT uf.timestamp, uf.feature_name, uf.feature_value
            FROM universal_features uf
            INNER JOIN (
                SELECT feature_name 
                FROM feature_metadata 
                ORDER BY importance_score DESC 
                LIMIT 10
            ) top_features ON uf.feature_name = top_features.feature_name
            """
            features_df = pd.read_sql(query, conn)
            
            # Перетворюємо в широкий формат
            features_pivot = features_df.pivot(
                index='timestamp', 
                columns='feature_name', 
                values='feature_value'
            )
            
            # Об'єднуємо з основними даними
            self.data = self.data.merge(
                features_pivot, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
        except:
            print("Додаткові фічі не завантажені")
        
        conn.close()
        
        print(f"Завантажено {len(self.data)} записів")
        print(f"Колонки: {list(self.data.columns)}")
        
        return self.data
    
    def auto_arima_selection(self, y, seasonal=False):
        """Автоматичний вибір параметрів ARIMA"""
        print("\n=== Auto-ARIMA Selection ===")
        
        # Тест на стаціонарність
        adf_result = adfuller(y.dropna())
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        
        # Auto ARIMA
        auto_model = auto_arima(
            y, 
            start_p=0, start_q=0, 
            max_p=5, max_q=5, 
            seasonal=seasonal,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=True,
            n_jobs=-1
        )
        
        print(f"\nНайкращі параметри: ARIMA{auto_model.order}")
        if seasonal:
            print(f"Сезонні параметри: {auto_model.seasonal_order}")
        
        return auto_model.order
    
    def create_ensemble_models(self, y, base_order):
        """Створення ансамблю моделей з різними параметрами"""
        print("\n=== Створення ансамблю ARIMA моделей ===")
        
        ensemble = {}
        base_p, base_d, base_q = base_order
        
        # Варіації параметрів
        param_variations = [
            (base_p, base_d, base_q),  # базова модель
            (max(0, base_p-1), base_d, base_q),
            (min(5, base_p+1), base_d, base_q),
            (base_p, base_d, max(0, base_q-1)),
            (base_p, base_d, min(5, base_q+1)),
            (max(0, base_p-1), base_d, max(0, base_q-1)),
            (min(5, base_p+1), base_d, min(5, base_q+1))
        ]
        
        # Видаляємо дублікати
        param_variations = list(set(param_variations))
        
        for params in param_variations:
            try:
                model = ARIMA(y, order=params)
                fitted_model = model.fit()
                
                # Оцінка якості на останніх 24 годинах
                train_size = len(y) - 24
                train, test = y[:train_size], y[train_size:]
                
                # Прогноз
                forecast = fitted_model.forecast(steps=24)
                mape = mean_absolute_percentage_error(test, forecast)
                
                ensemble[params] = {
                    'model': fitted_model,
                    'mape': mape,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
                
                print(f"ARIMA{params}: MAPE={mape:.2%}, AIC={fitted_model.aic:.2f}")
                
            except Exception as e:
                print(f"ARIMA{params}: Помилка - {str(e)[:50]}")
                continue
        
        return ensemble
    
    def rolling_window_retrain(self, target_column='btc_price'):
        """Rolling window retraining"""
        print("\n=== Rolling Window Retraining ===")
        
        y = self.data[target_column]
        results = []
        
        # Визначаємо точки перенавчання
        retrain_points = range(
            self.window_size, 
            len(y) - self.retrain_interval, 
            self.retrain_interval
        )
        
        for i, train_end in enumerate(retrain_points):
            if i % 5 == 0:  # Прогрес кожні 5 ітерацій
                print(f"\nRetraining iteration {i+1}/{len(retrain_points)}")
            
            # Вікно для навчання
            train_start = max(0, train_end - self.window_size)
            train_data = y[train_start:train_end]
            
            # Автоматичний вибір параметрів
            best_order = self.auto_arima_selection(train_data)
            
            # Створення ансамблю
            ensemble = self.create_ensemble_models(train_data, best_order)
            
            # Вибір найкращої моделі
            best_model_params = min(ensemble.keys(), key=lambda x: ensemble[x]['mape'])
            best_model = ensemble[best_model_params]['model']
            
            # Прогноз на наступний інтервал
            forecast_steps = min(self.retrain_interval, len(y) - train_end)
            forecast = best_model.forecast(steps=forecast_steps)
            actual = y[train_end:train_end + forecast_steps]
            
            # Оцінка
            if len(actual) > 0 and len(forecast) > 0:
                mape = mean_absolute_percentage_error(actual, forecast)
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                
                # Напрямок (для точності >80%)
                direction_pred = (forecast.values > train_data.iloc[-1]).astype(int)
                direction_actual = (actual.values > train_data.iloc[-1]).astype(int)
                direction_accuracy = (direction_pred == direction_actual).mean()
                
                results.append({
                    'timestamp': y.index[train_end],
                    'model_params': best_model_params,
                    'mape': mape,
                    'rmse': rmse,
                    'direction_accuracy': direction_accuracy,
                    'forecast_horizon': forecast_steps
                })
                
                # Зберігаємо модель
                self.models[y.index[train_end]] = {
                    'model': best_model,
                    'params': best_model_params,
                    'ensemble': ensemble
                }
        
        self.performance_history = pd.DataFrame(results)
        
        print(f"\n=== Підсумок Rolling Window ===")
        print(f"Всього перенавчань: {len(results)}")
        print(f"Середня MAPE: {self.performance_history['mape'].mean():.2%}")
        print(f"Середня точність напрямку: {self.performance_history['direction_accuracy'].mean():.2%}")
        
        return self.performance_history
    
    def adaptive_model_switching(self):
        """Адаптивне перемикання між моделями"""
        print("\n=== Adaptive Model Switching ===")
        
        # Аналіз performance history
        if len(self.performance_history) == 0:
            return
        
        # Групуємо по параметрах моделі
        param_performance = self.performance_history.groupby('model_params').agg({
            'mape': 'mean',
            'direction_accuracy': 'mean',
            'rmse': 'mean'
        }).reset_index()
        
        print("\nПродуктивність різних конфігурацій:")
        for _, row in param_performance.iterrows():
            print(f"ARIMA{row['model_params']}: "
                  f"MAPE={row['mape']:.2%}, "
                  f"Direction={row['direction_accuracy']:.2%}")
        
        # Визначаємо правила перемикання
        switching_rules = {
            'high_volatility': (2, 1, 2),  # Більше q для волатильності
            'trend': (1, 2, 1),  # Більше d для тренду
            'stable': (1, 1, 1),  # Проста модель для стабільності
            'complex': (3, 1, 3)  # Складна модель
        }
        
        # Визначаємо поточний режим ринку
        recent_volatility = self.data['btc_price'].pct_change().rolling(24).std().iloc[-1]
        recent_trend = self.data['btc_price'].diff().rolling(24).mean().iloc[-1]
        
        if recent_volatility > 0.02:
            market_regime = 'high_volatility'
        elif abs(recent_trend) > 100:
            market_regime = 'trend'
        elif recent_volatility < 0.01:
            market_regime = 'stable'
        else:
            market_regime = 'complex'
        
        recommended_params = switching_rules[market_regime]
        
        print(f"\nПоточний режим ринку: {market_regime}")
        print(f"Рекомендовані параметри: ARIMA{recommended_params}")
        
        self.best_params = {
            'market_regime': market_regime,
            'recommended_params': recommended_params,
            'recent_volatility': recent_volatility,
            'recent_trend': recent_trend
        }
        
        return self.best_params
    
    def ensemble_forecasting(self, horizon=24):
        """Ансамблеве прогнозування"""
        print(f"\n=== Ensemble Forecasting (horizon={horizon}h) ===")
        
        # Беремо останню модель
        if not self.models:
            print("Немає навчених моделей!")
            return None
        
        latest_timestamp = max(self.models.keys())
        latest_models = self.models[latest_timestamp]['ensemble']
        
        forecasts = {}
        weights = {}
        
        # Генеруємо прогнози від кожної моделі
        for params, model_info in latest_models.items():
            try:
                forecast = model_info['model'].forecast(steps=horizon)
                forecasts[params] = forecast
                
                # Ваги на основі MAPE (інверсія)
                weights[params] = 1 / (model_info['mape'] + 0.001)
            except:
                continue
        
        if not forecasts:
            return None
        
        # Нормалізація ваг
        total_weight = sum(weights.values())
        for params in weights:
            weights[params] /= total_weight
        
        # Зважений ансамблевий прогноз
        ensemble_forecast = pd.Series(index=forecasts[list(forecasts.keys())[0]].index, 
                                    data=0.0)
        
        for params, forecast in forecasts.items():
            ensemble_forecast += forecast * weights[params]
        
        print(f"\nВаги ансамблю:")
        for params, weight in weights.items():
            print(f"  ARIMA{params}: {weight:.2%}")
        
        self.ensemble_weights = weights
        
        return ensemble_forecast
    
    def save_results(self):
        """Збереження результатів"""
        print("\n=== Збереження результатів ===")
        
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Створюємо таблицю для adaptive ARIMA
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS adaptive_arima_results (
            timestamp DATETIME,
            model_params VARCHAR(50),
            mape DECIMAL(10,6),
            direction_accuracy DECIMAL(10,6),
            rmse DECIMAL(20,6),
            market_regime VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (timestamp, model_params)
        )
        """)
        
        # Зберігаємо результати
        for _, row in self.performance_history.iterrows():
            cursor.execute("""
            INSERT INTO adaptive_arima_results 
            (timestamp, model_params, mape, direction_accuracy, rmse, market_regime)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            mape = VALUES(mape),
            direction_accuracy = VALUES(direction_accuracy),
            rmse = VALUES(rmse)
            """, (
                row['timestamp'],
                str(row['model_params']),
                float(row['mape']),
                float(row['direction_accuracy']),
                float(row['rmse']),
                self.best_params.get('market_regime', 'unknown')
            ))
        
        conn.commit()
        conn.close()
        
        # Зберігаємо в CSV
        self.performance_history.to_csv('adaptive_arima_performance.csv', index=False)
        
        # Зберігаємо параметри
        with open('adaptive_arima_params.json', 'w') as f:
            json.dump({
                'best_params': str(self.best_params),
                'ensemble_weights': {str(k): v for k, v in self.ensemble_weights.items()},
                'retrain_interval': self.retrain_interval,
                'window_size': self.window_size
            }, f, indent=2)
        
        print("✓ Результати збережено в БД")
        print("✓ adaptive_arima_performance.csv")
        print("✓ adaptive_arima_params.json")
    
    def visualize_performance(self):
        """Візуалізація продуктивності"""
        print("\n=== Візуалізація ===")
        
        if len(self.performance_history) == 0:
            print("Немає даних для візуалізації")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Adaptive ARIMA Performance', fontsize=16)
        
        # 1. MAPE over time
        ax = axes[0, 0]
        self.performance_history.plot(x='timestamp', y='mape', ax=ax)
        ax.set_title('MAPE протягом часу')
        ax.set_ylabel('MAPE')
        ax.axhline(y=0.12, color='r', linestyle='--', label='Цільова MAPE (12%)')
        ax.legend()
        
        # 2. Direction Accuracy
        ax = axes[0, 1]
        self.performance_history.plot(x='timestamp', y='direction_accuracy', ax=ax)
        ax.set_title('Точність прогнозу напрямку')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.8, color='g', linestyle='--', label='Ціль >80%')
        ax.legend()
        
        # 3. Model parameters distribution
        ax = axes[1, 0]
        param_counts = self.performance_history['model_params'].value_counts()
        param_counts.plot(kind='bar', ax=ax)
        ax.set_title('Розподіл використаних параметрів')
        ax.set_xlabel('ARIMA(p,d,q)')
        ax.set_ylabel('Кількість')
        
        # 4. Performance by parameters
        ax = axes[1, 1]
        param_perf = self.performance_history.groupby('model_params')['direction_accuracy'].mean()
        param_perf.plot(kind='bar', ax=ax)
        ax.set_title('Середня точність по параметрах')
        ax.set_xlabel('ARIMA(p,d,q)')
        ax.set_ylabel('Direction Accuracy')
        ax.axhline(y=0.8, color='g', linestyle='--')
        
        plt.tight_layout()
        plt.savefig('adaptive_arima_performance.png', dpi=300, bbox_inches='tight')
        print("✓ Збережено: adaptive_arima_performance.png")
        plt.close()
    
    def run_full_pipeline(self):
        """Запуск повного pipeline"""
        print("=== ADAPTIVE ARIMA FRAMEWORK ===\n")
        
        # 1. Завантаження даних
        self.load_data()
        
        # 2. Rolling window retraining
        self.rolling_window_retrain()
        
        # 3. Adaptive model switching
        self.adaptive_model_switching()
        
        # 4. Ensemble forecasting
        forecast = self.ensemble_forecasting(horizon=24)
        
        if forecast is not None:
            print(f"\nПрогноз на наступні 24 години:")
            print(f"Мін: ${forecast.min():.2f}")
            print(f"Макс: ${forecast.max():.2f}")
            print(f"Середнє: ${forecast.mean():.2f}")
        
        # 5. Візуалізація
        self.visualize_performance()
        
        # 6. Збереження
        self.save_results()
        
        # Підсумок
        print("\n=== ПІДСУМОК ===")
        avg_mape = self.performance_history['mape'].mean()
        avg_direction = self.performance_history['direction_accuracy'].mean()
        
        print(f"Середня MAPE: {avg_mape:.2%}")
        print(f"Середня точність напрямку: {avg_direction:.2%}")
        
        if avg_direction > 0.8:
            print("✅ ЦІЛЬ ДОСЯГНУТА: Точність напрямку >80%")
        else:
            print(f"⚠️ Точність напрямку {avg_direction:.2%} < 80%")
        
        return {
            'avg_mape': avg_mape,
            'avg_direction_accuracy': avg_direction,
            'best_params': self.best_params,
            'models_count': len(self.models)
        }


def main():
    framework = AdaptiveARIMAFramework()
    results = framework.run_full_pipeline()
    
    print("\n✅ Adaptive ARIMA Framework успішно виконано!")
    print(f"Навчено моделей: {results['models_count']}")
    print(f"Найкращі параметри: {results['best_params']['recommended_params']}")


if __name__ == "__main__":
    main()