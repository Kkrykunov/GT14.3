#!/usr/bin/env python3
"""
ЕТАП 3: Оптимізована самонавчальна ARIMA з швидкою конвергенцією
Цільова точність: 12% MAPE
"""

import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import json

class OptimizedSelfLearningARIMA:
    """Оптимізована самонавчальна ARIMA модель"""
    
    def __init__(self):
        self.config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        
        self.target_mape = 12.0
        self.best_mape = float('inf')
        
    def run_optimized_arima(self):
        """Запуск оптимізованої ARIMA"""
        print(" ЕТАП 3: ОПТИМІЗОВАНА САМОНАВЧАЛЬНА ARIMA")
        print("=" * 60)
        print(f" Цільова точність: {self.target_mape}% MAPE")
        print()
        
        # 1. Завантаження та підготовка даних
        df = self.load_data()
        price_series = self.prepare_price_series(df)
        
        # 2. Швидка оптимізація
        best_params = self.quick_optimization(price_series)
        
        # 3. Навчання та валідація
        model_results = self.train_and_validate(price_series, best_params)
        
        # 4. Генерація прогнозів
        forecasts = self.generate_forecasts(price_series, best_params)
        
        # 5. Збереження результатів
        self.save_results(best_params, model_results, forecasts)
        
        return {
            'best_params': best_params,
            'accuracy_metrics': model_results,
            'forecasts': forecasts,
            'target_achieved': model_results['mape'] <= self.target_mape
        }
    
    def load_data(self):
        """Завантаження даних"""
        print(" ЗАВАНТАЖЕННЯ ДАНИХ...")
        
        conn = mysql.connector.connect(**self.config)
        
        query = """
        SELECT timestamp, btc_price
        FROM whale_hourly_complete
        WHERE whale_activity > 0 AND btc_price > 0
        ORDER BY timestamp
        """
        
        df = pd.read_sql(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        conn.close()
        
        print(f" Завантажено {len(df)} записів")
        return df
    
    def prepare_price_series(self, df):
        """Підготовка часового ряду"""
        print(" ПІДГОТОВКА ЧАСОВОГО РЯДУ...")
        
        # Логарифмічне перетворення
        price_series = np.log(df['btc_price'].dropna())
        
        # Ресамплюємо до годинної частоти для забезпечення регулярності
        price_series = price_series.resample('H').last().dropna()
        
        print(f" Підготовлено ряд з {len(price_series)} спостережень")
        return price_series
    
    def quick_optimization(self, series):
        """Швидка оптимізація параметрів"""
        print(" ШВИДКА ОПТИМІЗАЦІЯ ПАРАМЕТРІВ...")
        
        # Тест на стаціонарність
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] <= 0.05
        
        print(f"ADF p-value: {adf_result[1]:.4f} ({'Стаціонарний' if is_stationary else 'Нестаціонарний'})")
        
        # Розділяємо дані
        train_size = int(len(series) * 0.8)
        train_series = series[:train_size]
        test_series = series[train_size:]
        
        # Попередньо відібрані оптимальні параметри для фінансових рядів
        candidate_models = [
            (1, 1, 1),  # Класична ARIMA(1,1,1)
            (2, 1, 1),  # AR(2)
            (1, 1, 2),  # MA(2)
            (2, 1, 2),  # ARIMA(2,1,2)
            (0, 1, 1),  # ARIMA(0,1,1) = random walk
            (1, 0, 1),  # ARIMA(1,0,1) для стаціонарних рядів
        ]
        
        best_mape = float('inf')
        best_order = None
        
        for order in candidate_models:
            try:
                model = ARIMA(train_series, order=order)
                fitted_model = model.fit()
                
                # Прогноз на тестовій вибірці
                forecast = fitted_model.forecast(steps=len(test_series))
                
                # Конвертуємо назад з логарифмічної шкали
                actual_prices = np.exp(test_series)
                forecast_prices = np.exp(forecast)
                
                mape = mean_absolute_percentage_error(actual_prices, forecast_prices) * 100
                
                if mape < best_mape:
                    best_mape = mape
                    best_order = order
                
                print(f"   ARIMA{order}: MAPE = {mape:.2f}%")
                
            except Exception as e:
                print(f"   ARIMA{order}: Помилка")
                continue
        
        print(f"\n Найкраща модель: ARIMA{best_order}")
        print(f" MAPE: {best_mape:.2f}%")
        
        return {
            'order': best_order,
            'mape': best_mape
        }
    
    def train_and_validate(self, series, best_params):
        """Навчання та валідація моделі"""
        print("\n НАВЧАННЯ ТА ВАЛІДАЦІЯ...")
        
        # Навчання на повному наборі даних
        model = ARIMA(series, order=best_params['order'])
        fitted_model = model.fit()
        
        # Крос-валідація з ковзним вікном
        mape_scores = []
        mae_scores = []
        window_size = 168  # Тиждень
        forecast_horizon = 24  # 24 години
        
        for i in range(len(series) - window_size - forecast_horizon, len(series) - forecast_horizon, 24):
            try:
                train_data = series[i:i + window_size]
                test_data = series[i + window_size:i + window_size + forecast_horizon]
                
                model_cv = ARIMA(train_data, order=best_params['order'])
                fitted_cv = model_cv.fit()
                
                forecast_cv = fitted_cv.forecast(steps=len(test_data))
                
                # Конвертуємо з логарифмічної шкали
                actual_prices = np.exp(test_data)
                forecast_prices = np.exp(forecast_cv)
                
                mape = mean_absolute_percentage_error(actual_prices, forecast_prices) * 100
                mae = mean_absolute_error(actual_prices, forecast_prices)
                
                mape_scores.append(mape)
                mae_scores.append(mae)
                
            except:
                continue
        
        avg_mape = np.mean(mape_scores)
        avg_mae = np.mean(mae_scores)
        
        # Діагностика моделі
        residuals = fitted_model.resid
        
        results = {
            'mape': avg_mape,
            'mae': avg_mae,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'target_achieved': avg_mape <= self.target_mape
        }
        
        print(f" РЕЗУЛЬТАТИ ВАЛІДАЦІЇ:")
        print(f"   MAPE: {avg_mape:.2f}% ({'' if avg_mape <= self.target_mape else ''} Ціль: {self.target_mape}%)")
        print(f"   MAE: ${avg_mae:.2f}")
        print(f"   AIC: {fitted_model.aic:.2f}")
        print(f"   BIC: {fitted_model.bic:.2f}")
        
        if avg_mape <= self.target_mape:
            print(f" ЦІЛЬ ДОСЯГНУТА! MAPE = {avg_mape:.2f}% ≤ {self.target_mape}%")
        else:
            print(f" Ціль не досягнута. Потрібно покращення.")
        
        self.fitted_model = fitted_model
        return results
    
    def generate_forecasts(self, series, best_params, periods=24):
        """Генерація прогнозів"""
        print(f"\n ГЕНЕРАЦІЯ ПРОГНОЗІВ НА {periods} ГОДИН...")
        
        # Прогноз
        forecast = self.fitted_model.forecast(steps=periods)
        confidence_intervals = self.fitted_model.get_forecast(steps=periods).conf_int()
        
        # Конвертуємо з логарифмічної шкали
        forecast_prices = np.exp(forecast)
        ci_lower = np.exp(confidence_intervals.iloc[:, 0])
        ci_upper = np.exp(confidence_intervals.iloc[:, 1])
        
        # Створюємо часові мітки
        last_timestamp = series.index[-1]
        forecast_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=periods,
            freq='H'
        )
        
        forecasts_df = pd.DataFrame({
            'timestamp': forecast_timestamps,
            'forecast_price': forecast_prices,
            'lower_ci': ci_lower,
            'upper_ci': ci_upper,
            'confidence_width': ci_upper - ci_lower
        })
        
        current_price = np.exp(series.iloc[-1])
        forecast_1h = forecast_prices.iloc[0]
        forecast_24h = forecast_prices.iloc[-1]
        
        print(f" Прогнози згенеровані:")
        print(f"   Поточна ціна: ${current_price:,.0f}")
        print(f"   Прогноз на +1h: ${forecast_1h:,.0f} ({((forecast_1h/current_price-1)*100):+.1f}%)")
        print(f"   Прогноз на +24h: ${forecast_24h:,.0f} ({((forecast_24h/current_price-1)*100):+.1f}%)")
        
        return forecasts_df
    
    def save_results(self, best_params, model_results, forecasts):
        """Збереження результатів"""
        print("\n ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ...")
        
        conn = mysql.connector.connect(**self.config)
        cursor = conn.cursor()
        
        # Таблиця моделей
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS arima_models (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100),
            order_p INT,
            order_d INT,
            order_q INT,
            mape DECIMAL(10,4),
            mae DECIMAL(15,2),
            aic DECIMAL(15,6),
            bic DECIMAL(15,6),
            target_achieved BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Очищуємо старі записи
        cursor.execute("DELETE FROM arima_models")
        
        # Вставка моделі
        cursor.execute("""
        INSERT INTO arima_models 
        (model_name, order_p, order_d, order_q, mape, mae, aic, bic, target_achieved)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            f"ARIMA{best_params['order']}",
            int(best_params['order'][0]), int(best_params['order'][1]), int(best_params['order'][2]),
            float(model_results['mape']), float(model_results['mae']),
            float(model_results['aic']), float(model_results['bic']),
            bool(model_results['target_achieved'])
        ))
        
        # Таблиця прогнозів
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS arima_forecasts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            forecast_timestamp DATETIME,
            forecast_price DECIMAL(15,2),
            lower_ci DECIMAL(15,2),
            upper_ci DECIMAL(15,2),
            confidence_width DECIMAL(15,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_forecast_timestamp (forecast_timestamp)
        )
        """)
        
        # Очищуємо старі прогнози
        cursor.execute("DELETE FROM arima_forecasts")
        
        # Вставка прогнозів
        for _, row in forecasts.iterrows():
            cursor.execute("""
            INSERT INTO arima_forecasts 
            (forecast_timestamp, forecast_price, lower_ci, upper_ci, confidence_width)
            VALUES (%s, %s, %s, %s, %s)
            """, (
                row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                float(row['forecast_price']),
                float(row['lower_ci']),
                float(row['upper_ci']),
                float(row['confidence_width'])
            ))
        
        conn.commit()
        conn.close()
        
        print(f" Збережено модель та {len(forecasts)} прогнозів")

def main():
    """Головна функція"""
    arima_engine = OptimizedSelfLearningARIMA()
    
    results = arima_engine.run_optimized_arima()
    
    print(f"\n ЕТАП 3 ЗАВЕРШЕНО!")
    print(f"Модель: ARIMA{results['best_params']['order']}")
    print(f"MAPE: {results['accuracy_metrics']['mape']:.2f}%")
    print(f"Ціль досягнута: {' ТАК' if results['target_achieved'] else ' НІ'}")

if __name__ == "__main__":
    main()