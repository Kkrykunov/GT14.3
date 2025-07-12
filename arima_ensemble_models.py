#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA Ensemble Models - Розширення до 8 моделей
Включає ARIMA, SARIMA варіанти та ансамблеве прогнозування
"""

import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import logging
import json
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARIMAEnsembleModels:
    """Клас для роботи з ансамблем ARIMA моделей"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        
        # Конфігурація 8 моделей
        self.models_config = {
            'ARIMA_101': {'order': (1, 0, 1), 'type': 'ARIMA'},
            'ARIMA_111': {'order': (1, 1, 1), 'type': 'ARIMA'},
            'ARIMA_211': {'order': (2, 1, 1), 'type': 'ARIMA'},
            'ARIMA_212': {'order': (2, 1, 2), 'type': 'ARIMA'},
            'SARIMA_111_111_24': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 24),
                'type': 'SARIMA'
            },
            'SARIMA_101_110_24': {
                'order': (1, 0, 1),
                'seasonal_order': (1, 1, 0, 24),
                'type': 'SARIMA'
            },
            'HoltWinters_add': {
                'seasonal': 'add',
                'seasonal_periods': 24,
                'type': 'HoltWinters'
            },
            'HoltWinters_mul': {
                'seasonal': 'mul',
                'seasonal_periods': 24,
                'type': 'HoltWinters'
            }
        }
        
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        
    def load_data(self, hours=168):
        """Завантаження даних для моделювання"""
        logger.info(f"Завантаження даних за останні {hours} годин")
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            query = """
            SELECT timestamp, btc_price
            FROM whale_hourly_complete
            WHERE btc_price > 0
            ORDER BY timestamp DESC
            LIMIT %s
            """
            
            df = pd.read_sql(query, conn, params=(hours,))
            conn.close()
            
            # Сортування та підготовка
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Завантажено {len(df)} записів")
            logger.info(f"Період: {df.index.min()} - {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Помилка завантаження даних: {str(e)}")
            raise
            
    def check_stationarity(self, series):
        """Перевірка стаціонарності часового ряду"""
        result = adfuller(series.dropna())
        return result[1] < 0.05  # p-value < 0.05 означає стаціонарність
        
    def fit_single_model(self, train_data, model_name, config):
        """Навчання однієї моделі"""
        logger.info(f"Навчання моделі {model_name}")
        
        try:
            if config['type'] == 'ARIMA':
                model = ARIMA(train_data, order=config['order'])
                fitted = model.fit()
                
            elif config['type'] == 'SARIMA':
                model = SARIMAX(
                    train_data,
                    order=config['order'],
                    seasonal_order=config['seasonal_order'],
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted = model.fit(disp=False)
                
            elif config['type'] == 'HoltWinters':
                # Перевірка на додатні значення для мультиплікативної моделі
                if config['seasonal'] == 'mul' and (train_data <= 0).any():
                    logger.warning(f"{model_name}: Пропуск через негативні значення")
                    return None
                    
                model = ExponentialSmoothing(
                    train_data,
                    seasonal=config['seasonal'],
                    seasonal_periods=config['seasonal_periods']
                )
                fitted = model.fit()
                
            else:
                raise ValueError(f"Невідомий тип моделі: {config['type']}")
                
            # Метрики на тренувальних даних
            if hasattr(fitted, 'aic'):
                aic = fitted.aic
            else:
                aic = None
                
            logger.info(f"{model_name} навчено успішно. AIC: {aic}")
            return fitted
            
        except Exception as e:
            logger.error(f"Помилка навчання {model_name}: {str(e)}")
            return None
            
    def generate_forecasts(self, train_data, test_data, horizon=24):
        """Генерація прогнозів для всіх моделей"""
        logger.info(f"Генерація прогнозів на {horizon} годин вперед")
        
        results = {}
        
        for model_name, config in self.models_config.items():
            logger.info(f"\nМодель: {model_name}")
            
            # Навчання моделі
            fitted_model = self.fit_single_model(train_data, model_name, config)
            
            if fitted_model is None:
                continue
                
            try:
                # Прогнозування
                if config['type'] in ['ARIMA', 'SARIMA']:
                    forecast = fitted_model.forecast(steps=horizon)
                else:  # HoltWinters
                    forecast = fitted_model.forecast(steps=horizon)
                    
                # Розрахунок метрик якщо є тестові дані
                if len(test_data) >= horizon:
                    actual = test_data.iloc[:horizon]
                    mape = mean_absolute_percentage_error(actual, forecast) * 100
                    rmse = np.sqrt(mean_squared_error(actual, forecast))
                    mae = np.mean(np.abs(actual - forecast))
                else:
                    mape = rmse = mae = None
                    
                results[model_name] = {
                    'model': fitted_model,
                    'forecast': forecast,
                    'metrics': {
                        'mape': mape,
                        'rmse': rmse,
                        'mae': mae,
                        'aic': fitted_model.aic if hasattr(fitted_model, 'aic') else None
                    },
                    'config': config
                }
                
                if mape:
                    logger.info(f"MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")
                    
            except Exception as e:
                logger.error(f"Помилка прогнозування {model_name}: {str(e)}")
                continue
                
        self.models = results
        return results
        
    def create_ensemble_forecast(self, method='weighted_average'):
        """Створення ансамблевого прогнозу"""
        logger.info(f"Створення ансамблевого прогнозу методом: {method}")
        
        if not self.models:
            raise ValueError("Немає навчених моделей для ансамблю")
            
        # Збираємо всі прогнози
        forecasts = []
        weights = []
        
        for model_name, result in self.models.items():
            if 'forecast' in result:
                forecasts.append(result['forecast'])
                
                # Ваги на основі MAPE (менше краще)
                if method == 'weighted_average' and result['metrics']['mape']:
                    weight = 1 / result['metrics']['mape']
                else:
                    weight = 1
                    
                weights.append(weight)
                
        if not forecasts:
            raise ValueError("Немає доступних прогнозів")
            
        # Конвертація в DataFrame для зручності
        forecasts_df = pd.DataFrame(forecasts).T
        
        if method == 'simple_average':
            ensemble_forecast = forecasts_df.mean(axis=1)
            
        elif method == 'weighted_average':
            # Нормалізація ваг
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_forecast = (forecasts_df * weights).sum(axis=1)
            
        elif method == 'median':
            ensemble_forecast = forecasts_df.median(axis=1)
            
        else:
            raise ValueError(f"Невідомий метод ансамблю: {method}")
            
        return ensemble_forecast
        
    def visualize_model_comparison(self, train_data, test_data=None, save_path=None):
        """Візуалізація порівняння всіх моделей"""
        logger.info("Створення візуалізації порівняння моделей")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        # Історичні дані
        historical = train_data[-48:]  # Останні 48 годин
        
        for idx, (model_name, result) in enumerate(self.models.items()):
            if idx >= 9:
                break
                
            ax = axes[idx]
            
            # Історичні дані
            ax.plot(historical.index, historical.values, 'b-', label='Історичні дані', alpha=0.7)
            
            # Прогноз
            forecast = result['forecast']
            forecast_index = pd.date_range(
                start=historical.index[-1] + pd.Timedelta(hours=1),
                periods=len(forecast),
                freq='H'
            )
            ax.plot(forecast_index, forecast.values, 'r-', label='Прогноз', linewidth=2)
            
            # Тестові дані якщо є
            if test_data is not None and len(test_data) > 0:
                test_plot = test_data[:len(forecast)]
                ax.plot(test_plot.index, test_plot.values, 'g--', 
                       label='Фактичні дані', alpha=0.7)
                
            # Метрики
            if result['metrics']['mape']:
                ax.text(0.02, 0.98, f"MAPE: {result['metrics']['mape']:.2f}%",
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                       
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Час')
            ax.set_ylabel('BTC Price')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
        # Ансамблевий прогноз в останній ячейці
        ax = axes[8]
        ensemble_forecast = self.create_ensemble_forecast()
        
        ax.plot(historical.index, historical.values, 'b-', label='Історичні дані', alpha=0.7)
        forecast_index = pd.date_range(
            start=historical.index[-1] + pd.Timedelta(hours=1),
            periods=len(ensemble_forecast),
            freq='H'
        )
        ax.plot(forecast_index, ensemble_forecast.values, 'purple', 
               label='Ансамблевий прогноз', linewidth=3)
               
        if test_data is not None and len(test_data) > 0:
            test_plot = test_data[:len(ensemble_forecast)]
            ax.plot(test_plot.index, test_plot.values, 'g--', 
                   label='Фактичні дані', alpha=0.7)
            
            # MAPE для ансамблю
            ensemble_mape = mean_absolute_percentage_error(
                test_plot.values, ensemble_forecast.values
            ) * 100
            ax.text(0.02, 0.98, f"MAPE: {ensemble_mape:.2f}%",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                   
        ax.set_title('ENSEMBLE FORECAST', fontsize=14, fontweight='bold', color='purple')
        ax.set_xlabel('Час')
        ax.set_ylabel('BTC Price')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('ARIMA Ensemble Models - Порівняння 8 моделей', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Візуалізація збережена: {save_path}")
            
        return fig
        
    def save_results(self):
        """Збереження результатів в БД та файли"""
        logger.info("Збереження результатів")
        
        # Підготовка даних для збереження
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'models_count': len(self.models),
            'models': {}
        }
        
        best_model = None
        best_mape = float('inf')
        
        for model_name, result in self.models.items():
            model_summary = {
                'type': result['config']['type'],
                'config': result['config'],
                'metrics': result['metrics'],
                'forecast_values': result['forecast'].tolist() if 'forecast' in result else []
            }
            
            results_summary['models'][model_name] = model_summary
            
            # Визначення найкращої моделі
            if result['metrics']['mape'] and result['metrics']['mape'] < best_mape:
                best_mape = result['metrics']['mape']
                best_model = model_name
                
        # Ансамблевий прогноз
        ensemble_forecast = self.create_ensemble_forecast()
        results_summary['ensemble_forecast'] = ensemble_forecast.tolist()
        results_summary['best_model'] = best_model
        results_summary['best_mape'] = best_mape
        
        # Збереження в JSON
        with open('arima_ensemble_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        logger.info(f"Результати збережено. Найкраща модель: {best_model} (MAPE: {best_mape:.2f}%)")
        
        return results_summary
        
    def generate_report(self):
        """Генерація звіту про результати"""
        logger.info("Генерація звіту")
        
        report = "# ARIMA Ensemble Models - Звіт\n\n"
        report += f"**Дата аналізу:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Результати по моделях\n\n"
        report += "| Модель | Тип | MAPE (%) | RMSE | MAE | AIC |\n"
        report += "|--------|-----|----------|------|-----|-----|\n"
        
        models_sorted = sorted(
            self.models.items(),
            key=lambda x: x[1]['metrics']['mape'] if x[1]['metrics']['mape'] else float('inf')
        )
        
        for model_name, result in models_sorted:
            m = result['metrics']
            report += f"| {model_name} | {result['config']['type']} | "
            report += f"{m['mape']:.2f} | " if m['mape'] else "N/A | "
            report += f"{m['rmse']:.2f} | " if m['rmse'] else "N/A | "
            report += f"{m['mae']:.2f} | " if m['mae'] else "N/A | "
            report += f"{m['aic']:.2f} |\n" if m['aic'] else "N/A |\n"
            
        # Найкраща модель
        best_model = models_sorted[0] if models_sorted else None
        if best_model:
            report += f"\n## Найкраща модель: {best_model[0]}\n"
            report += f"- MAPE: {best_model[1]['metrics']['mape']:.2f}%\n"
            
        report += "\n## Конфігурації моделей\n\n"
        for model_name, config in self.models_config.items():
            report += f"### {model_name}\n"
            report += f"```python\n{json.dumps(config, indent=2)}\n```\n\n"
            
        # Збереження звіту
        with open('arima_ensemble_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info("Звіт збережено: arima_ensemble_report.md")
        
        return report


def run_ensemble_analysis(forecast_horizon=24):
    """Запуск повного ансамблевого аналізу"""
    logger.info("=== ЗАПУСК ARIMA ENSEMBLE ANALYSIS ===")
    
    try:
        # Ініціалізація
        ensemble = ARIMAEnsembleModels()
        
        # Завантаження даних
        data = ensemble.load_data(hours=336)  # 14 днів
        
        # Розділення на train/test
        train_size = len(data) - forecast_horizon
        train_data = data['btc_price'][:train_size]
        test_data = data['btc_price'][train_size:]
        
        logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Навчання моделей та прогнозування
        ensemble.generate_forecasts(train_data, test_data, horizon=forecast_horizon)
        
        # Візуалізація
        fig = ensemble.visualize_model_comparison(
            train_data, 
            test_data,
            save_path='arima_ensemble_comparison.png'
        )
        
        # Збереження результатів
        results = ensemble.save_results()
        
        # Генерація звіту
        report = ensemble.generate_report()
        
        logger.info("=== АНАЛІЗ ЗАВЕРШЕНО УСПІШНО ===")
        
        return ensemble, results
        
    except Exception as e:
        logger.error(f"Критична помилка: {str(e)}")
        raise


if __name__ == "__main__":
    ensemble, results = run_ensemble_analysis(forecast_horizon=24)
    print(f"\n✅ Аналіз завершено!")
    print(f"Найкраща модель: {results['best_model']} (MAPE: {results['best_mape']:.2f}%)")
    print(f"Кількість моделей: {results['models_count']}")