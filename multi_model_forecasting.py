#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мульти-модельне прогнозування для GT14 v14.2
Інтеграція ARIMA, VAR та Байєс для комплексного прогнозування
"""

import pandas as pd
import numpy as np
import mysql.connector
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MultiModelForecasting:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        self.horizons = [1, 6, 12, 24, 48]  # Години
        
    def load_data(self):
        """Завантаження даних з БД"""
        print("=== Завантаження даних ===")
        
        conn = mysql.connector.connect(**self.db_config)
        
        # Завантажуємо основні дані
        query = """
        SELECT timestamp, whale_volume_usd, whale_activity, 
               exchange_inflow, exchange_outflow, net_flow, 
               btc_price, fear_greed_index
        FROM whale_hourly_complete
        ORDER BY timestamp
        """
        
        df = pd.read_sql(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Завантажуємо кластерні мітки якщо є
        try:
            cluster_query = """
            SELECT timestamp, cluster_id 
            FROM cluster_labels
            ORDER BY timestamp
            """
            cluster_df = pd.read_sql(cluster_query, conn)
            cluster_df['timestamp'] = pd.to_datetime(cluster_df['timestamp'])
            cluster_df.set_index('timestamp', inplace=True)
            
            df = df.merge(cluster_df, left_index=True, right_index=True, how='left')
        except:
            print("Кластерні мітки не знайдено")
            df['cluster_id'] = 0
        
        conn.close()
        
        self.data = df
        print(f"Завантажено {len(df)} записів")
        
        return df
    
    def prepare_features(self):
        """Підготовка фічей для моделей"""
        print("\n=== Підготовка фічей ===")
        
        # Базові фічі
        self.features = ['whale_volume_usd', 'whale_activity', 
                        'exchange_inflow', 'exchange_outflow', 
                        'net_flow', 'fear_greed_index']
        
        # Додаємо технічні індикатори
        self.data['price_change'] = self.data['btc_price'].pct_change()
        self.data['volume_ma'] = self.data['whale_volume_usd'].rolling(24).mean()
        self.data['net_flow_ma'] = self.data['net_flow'].rolling(24).mean()
        
        # Додаємо лагові фічі
        for lag in [1, 6, 24]:
            self.data[f'price_lag_{lag}'] = self.data['btc_price'].shift(lag)
            self.data[f'volume_lag_{lag}'] = self.data['whale_volume_usd'].shift(lag)
        
        # Видаляємо NaN
        self.data = self.data.dropna()
        
        print(f"Підготовлено {len(self.features)} базових фічей")
    
    def train_arima_models(self):
        """Навчання ARIMA моделей для різних горизонтів"""
        print("\n=== Навчання ARIMA моделей ===")
        
        self.models['arima'] = {}
        
        for horizon in self.horizons:
            print(f"\nНавчання ARIMA для горизонту {horizon}h...")
            
            # Підготовка даних
            y = self.data['btc_price']
            
            # Train-test split
            split_point = int(len(y) * 0.8)
            train, test = y[:split_point], y[split_point:]
            
            # Пошук оптимальних параметрів
            best_aic = np.inf
            best_order = None
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(train, order=(p, d, q))
                            model_fit = model.fit()
                            
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Навчання фінальної моделі
            if best_order:
                model = ARIMA(train, order=best_order)
                model_fit = model.fit()
                self.models['arima'][horizon] = model_fit
                
                # Прогноз
                forecast = model_fit.forecast(steps=horizon)
                
                print(f"  ARIMA{best_order} - AIC: {best_aic:.2f}")
            else:
                print(f"  Не вдалося знайти оптимальні параметри")
    
    def train_var_models(self):
        """Навчання VAR моделей"""
        print("\n=== Навчання VAR моделей ===")
        
        self.models['var'] = {}
        
        # Підготовка даних для VAR
        var_features = ['whale_volume_usd', 'net_flow', 'btc_price', 
                       'whale_activity', 'fear_greed_index']
        
        for horizon in self.horizons:
            print(f"\nНавчання VAR для горизонту {horizon}h...")
            
            # Дані для VAR
            var_data = self.data[var_features].dropna()
            
            # Нормалізація
            scaler = StandardScaler()
            var_data_scaled = pd.DataFrame(
                scaler.fit_transform(var_data),
                index=var_data.index,
                columns=var_data.columns
            )
            
            # Train-test split
            split_point = int(len(var_data_scaled) * 0.8)
            train = var_data_scaled[:split_point]
            
            # Визначення оптимального лагу
            model = VAR(train)
            lag_order = model.select_order(maxlags=12)
            optimal_lag = lag_order.aic
            
            # Навчання моделі
            var_model = model.fit(optimal_lag)
            self.models['var'][horizon] = {
                'model': var_model,
                'scaler': scaler,
                'features': var_features
            }
            
            print(f"  VAR з лагом {optimal_lag}")
    
    def train_bayes_models(self):
        """Навчання Байєс моделей для класифікації напрямку"""
        print("\n=== Навчання Байєс моделей ===")
        
        self.models['bayes'] = {}
        
        for horizon in self.horizons:
            print(f"\nНавчання Байєс для горизонту {horizon}h...")
            
            # Підготовка даних
            X = self.data[self.features].values
            
            # Цільова змінна - напрямок руху ціни
            y = (self.data['btc_price'].shift(-horizon) > self.data['btc_price']).astype(int)
            y = y.dropna()
            X = X[:len(y)]
            
            # Train-test split
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Навчання моделі
            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)
            
            # Оцінка точності
            y_pred = nb_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models['bayes'][horizon] = nb_model
            
            print(f"  Точність: {accuracy:.2%}")
    
    def generate_forecasts(self, start_date=None, end_date=None):
        """Генерація прогнозів всіма моделями"""
        print("\n=== Генерація прогнозів ===")
        
        if not start_date:
            start_date = self.data.index[-100]
        if not end_date:
            end_date = self.data.index[-1]
        
        test_data = self.data[start_date:end_date]
        
        for horizon in self.horizons:
            print(f"\nПрогнози для горизонту {horizon}h:")
            self.forecasts[horizon] = {}
            
            # ARIMA прогноз
            if horizon in self.models.get('arima', {}):
                arima_model = self.models['arima'][horizon]
                arima_forecast = []
                
                for i in range(len(test_data) - horizon):
                    forecast = arima_model.forecast(steps=horizon)
                    arima_forecast.append(forecast.iloc[-1])
                
                self.forecasts[horizon]['arima'] = arima_forecast
                print(f"  ARIMA: {len(arima_forecast)} прогнозів")
            
            # VAR прогноз
            if horizon in self.models.get('var', {}):
                var_dict = self.models['var'][horizon]
                var_model = var_dict['model']
                scaler = var_dict['scaler']
                
                var_forecast = []
                
                for i in range(len(test_data) - horizon):
                    # Підготовка даних
                    current_data = test_data[var_dict['features']].iloc[i:i+1]
                    scaled_data = scaler.transform(current_data)
                    
                    # Прогноз
                    forecast = var_model.forecast(scaled_data, steps=horizon)
                    # Денормалізація
                    forecast_denorm = scaler.inverse_transform(forecast)
                    # Беремо прогноз ціни (індекс 2)
                    price_forecast = forecast_denorm[-1, 2]
                    var_forecast.append(price_forecast)
                
                self.forecasts[horizon]['var'] = var_forecast
                print(f"  VAR: {len(var_forecast)} прогнозів")
            
            # Байєс прогноз (напрямок)
            if horizon in self.models.get('bayes', {}):
                bayes_model = self.models['bayes'][horizon]
                bayes_forecast = []
                
                for i in range(len(test_data) - horizon):
                    X = test_data[self.features].iloc[i:i+1].values
                    # Ймовірність руху вгору
                    prob_up = bayes_model.predict_proba(X)[0, 1]
                    bayes_forecast.append(prob_up)
                
                self.forecasts[horizon]['bayes'] = bayes_forecast
                print(f"  Байєс: {len(bayes_forecast)} прогнозів")
    
    def evaluate_models(self):
        """Оцінка точності моделей"""
        print("\n=== Оцінка моделей ===")
        
        results = []
        
        for horizon in self.horizons:
            if horizon not in self.forecasts:
                continue
            
            print(f"\nГоризонт {horizon}h:")
            
            # Реальні значення
            actual_prices = self.data['btc_price'].iloc[-len(self.forecasts[horizon].get('arima', [])):].values
            actual_direction = (self.data['btc_price'].shift(-horizon) > self.data['btc_price']).iloc[-len(self.forecasts[horizon].get('arima', [])):].values
            
            # ARIMA оцінка
            if 'arima' in self.forecasts[horizon]:
                arima_pred = self.forecasts[horizon]['arima']
                if len(arima_pred) > 0:
                    mape = mean_absolute_percentage_error(actual_prices[:len(arima_pred)], arima_pred)
                    direction_acc = accuracy_score(
                        actual_direction[:len(arima_pred)],
                        [1 if p > actual_prices[i] else 0 for i, p in enumerate(arima_pred)]
                    )
                    
                    results.append({
                        'model': 'ARIMA',
                        'horizon': horizon,
                        'mape': mape,
                        'direction_accuracy': direction_acc
                    })
                    
                    print(f"  ARIMA - MAPE: {mape:.2%}, Direction Accuracy: {direction_acc:.2%}")
            
            # VAR оцінка
            if 'var' in self.forecasts[horizon]:
                var_pred = self.forecasts[horizon]['var']
                if len(var_pred) > 0:
                    mape = mean_absolute_percentage_error(actual_prices[:len(var_pred)], var_pred)
                    direction_acc = accuracy_score(
                        actual_direction[:len(var_pred)],
                        [1 if p > actual_prices[i] else 0 for i, p in enumerate(var_pred)]
                    )
                    
                    results.append({
                        'model': 'VAR',
                        'horizon': horizon,
                        'mape': mape,
                        'direction_accuracy': direction_acc
                    })
                    
                    print(f"  VAR - MAPE: {mape:.2%}, Direction Accuracy: {direction_acc:.2%}")
            
            # Байєс оцінка
            if 'bayes' in self.forecasts[horizon]:
                bayes_pred = self.forecasts[horizon]['bayes']
                if len(bayes_pred) > 0:
                    bayes_binary = [1 if p > 0.5 else 0 for p in bayes_pred]
                    direction_acc = accuracy_score(actual_direction[:len(bayes_pred)], bayes_binary)
                    
                    results.append({
                        'model': 'Bayes',
                        'horizon': horizon,
                        'mape': None,
                        'direction_accuracy': direction_acc
                    })
                    
                    print(f"  Байєс - Direction Accuracy: {direction_acc:.2%}")
        
        self.evaluation_results = pd.DataFrame(results)
        return self.evaluation_results
    
    def create_ensemble_forecast(self):
        """Створення ансамблевого прогнозу"""
        print("\n=== Створення ансамблевого прогнозу ===")
        
        ensemble_forecasts = {}
        
        for horizon in self.horizons:
            if horizon not in self.forecasts:
                continue
            
            # Визначення ваг на основі точності
            weights = {}
            total_weight = 0
            
            # Ваги на основі direction accuracy
            for model in ['arima', 'var']:
                if model in self.forecasts[horizon]:
                    model_results = self.evaluation_results[
                        (self.evaluation_results['model'] == model.upper()) & 
                        (self.evaluation_results['horizon'] == horizon)
                    ]
                    
                    if not model_results.empty:
                        accuracy = model_results['direction_accuracy'].values[0]
                        weights[model] = accuracy
                        total_weight += accuracy
            
            # Нормалізація ваг
            if total_weight > 0:
                for model in weights:
                    weights[model] /= total_weight
            
            # Створення ансамблевого прогнозу
            ensemble = []
            
            min_length = min(
                len(self.forecasts[horizon].get('arima', [])),
                len(self.forecasts[horizon].get('var', []))
            )
            
            for i in range(min_length):
                weighted_forecast = 0
                
                if 'arima' in weights:
                    weighted_forecast += weights['arima'] * self.forecasts[horizon]['arima'][i]
                if 'var' in weights:
                    weighted_forecast += weights['var'] * self.forecasts[horizon]['var'][i]
                
                ensemble.append(weighted_forecast)
            
            ensemble_forecasts[horizon] = ensemble
            
            print(f"Горизонт {horizon}h - Ваги: {weights}")
        
        return ensemble_forecasts
    
    def recommend_best_model(self):
        """Рекомендація найкращої моделі для кожного періоду"""
        print("\n=== Рекомендації по вибору моделі ===")
        
        recommendations = {}
        
        for horizon in self.horizons:
            # Фільтруємо результати для горизонту
            horizon_results = self.evaluation_results[
                self.evaluation_results['horizon'] == horizon
            ]
            
            if horizon_results.empty:
                continue
            
            # Найкраща модель за direction accuracy
            best_model = horizon_results.loc[
                horizon_results['direction_accuracy'].idxmax()
            ]
            
            recommendations[horizon] = {
                'best_model': best_model['model'],
                'accuracy': best_model['direction_accuracy'],
                'reason': self._get_recommendation_reason(horizon, best_model['model'])
            }
            
            print(f"\nГоризонт {horizon}h:")
            print(f"  Рекомендована модель: {best_model['model']}")
            print(f"  Точність: {best_model['direction_accuracy']:.2%}")
            print(f"  Причина: {recommendations[horizon]['reason']}")
        
        return recommendations
    
    def _get_recommendation_reason(self, horizon, model):
        """Пояснення вибору моделі"""
        reasons = {
            'ARIMA': {
                1: "ARIMA краще для короткострокових прогнозів через швидку адаптацію",
                6: "ARIMA ефективна для внутрішньоденних трендів",
                12: "ARIMA добре вловлює циклічні патерни",
                24: "ARIMA підходить для денних прогнозів",
                48: "ARIMA може втрачати точність на довгих горизонтах"
            },
            'VAR': {
                1: "VAR враховує взаємозв'язки між змінними",
                6: "VAR ефективна для середньострокових залежностей",
                12: "VAR добре моделює складні взаємодії",
                24: "VAR оптимальна для денного горизонту",
                48: "VAR краще для довгострокових структурних зв'язків"
            },
            'Bayes': {
                1: "Байєс фокусується на ймовірності напрямку",
                6: "Байєс надійна для бінарних прогнозів",
                12: "Байєс стабільна для класифікації тренду",
                24: "Байєс консервативна для денних рішень",
                48: "Байєс підходить для стратегічних позицій"
            }
        }
        
        return reasons.get(model, {}).get(horizon, "Модель показала найкращі результати")
    
    def visualize_comparison(self):
        """Візуалізація порівняння моделей"""
        print("\n=== Візуалізація результатів ===")
        
        # Створюємо фігуру
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Порівняння моделей прогнозування', fontsize=16)
        
        # 1. Точність по горизонтах
        ax = axes[0, 0]
        pivot_data = self.evaluation_results.pivot(
            index='horizon', 
            columns='model', 
            values='direction_accuracy'
        )
        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title('Точність прогнозу напрямку по горизонтах')
        ax.set_xlabel('Горизонт (години)')
        ax.set_ylabel('Точність (%)')
        ax.legend(title='Модель')
        
        # 2. Heatmap точності
        ax = axes[0, 1]
        sns.heatmap(pivot_data.T, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax)
        ax.set_title('Матриця точності моделей')
        
        # 3. MAPE для ARIMA та VAR
        ax = axes[1, 0]
        mape_data = self.evaluation_results[
            self.evaluation_results['mape'].notna()
        ].pivot(index='horizon', columns='model', values='mape')
        
        if not mape_data.empty:
            mape_data.plot(kind='line', marker='o', ax=ax)
            ax.set_title('MAPE по горизонтах')
            ax.set_xlabel('Горизонт (години)')
            ax.set_ylabel('MAPE (%)')
            ax.legend(title='Модель')
        
        # 4. Рекомендації
        ax = axes[1, 1]
        recommendations = self.recommend_best_model()
        
        rec_text = "Рекомендовані моделі:\n\n"
        for horizon, rec in recommendations.items():
            rec_text += f"{horizon}h: {rec['best_model']} ({rec['accuracy']:.1%})\n"
        
        ax.text(0.1, 0.5, rec_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('multi_model_comparison.png', dpi=300, bbox_inches='tight')
        print("Збережено: multi_model_comparison.png")
        
        plt.close()
    
    def save_results(self):
        """Збереження результатів"""
        print("\n=== Збереження результатів ===")
        
        # Збереження оцінок моделей
        self.evaluation_results.to_csv('model_evaluation_results.csv', index=False)
        print("✓ Збережено: model_evaluation_results.csv")
        
        # Збереження рекомендацій
        recommendations = self.recommend_best_model()
        rec_df = pd.DataFrame.from_dict(recommendations, orient='index')
        rec_df.to_csv('model_recommendations.csv')
        print("✓ Збережено: model_recommendations.csv")
        
        # Збереження в БД
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Створюємо таблицю для результатів
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            model_name VARCHAR(50),
            horizon INT,
            direction_accuracy DECIMAL(10,4),
            mape DECIMAL(10,4),
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (model_name, horizon)
        )
        """)
        
        # Вставляємо результати
        for _, row in self.evaluation_results.iterrows():
            cursor.execute("""
            INSERT INTO model_performance 
            (model_name, horizon, direction_accuracy, mape)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            direction_accuracy = VALUES(direction_accuracy),
            mape = VALUES(mape)
            """, (
                row['model'],
                int(row['horizon']),
                float(row['direction_accuracy']),
                float(row['mape']) if pd.notna(row['mape']) else None
            ))
        
        conn.commit()
        conn.close()
        
        print("✓ Результати збережено в БД")
    
    def run_full_pipeline(self):
        """Запуск повного pipeline"""
        print("=== МУЛЬТИ-МОДЕЛЬНЕ ПРОГНОЗУВАННЯ ===\n")
        
        # 1. Завантаження даних
        self.load_data()
        
        # 2. Підготовка фічей
        self.prepare_features()
        
        # 3. Навчання моделей
        self.train_arima_models()
        self.train_var_models()
        self.train_bayes_models()
        
        # 4. Генерація прогнозів
        self.generate_forecasts()
        
        # 5. Оцінка моделей
        self.evaluate_models()
        
        # 6. Створення ансамблю
        ensemble = self.create_ensemble_forecast()
        
        # 7. Візуалізація
        self.visualize_comparison()
        
        # 8. Збереження результатів
        self.save_results()
        
        print("\n=== АНАЛІЗ ЗАВЕРШЕНО ===")
        
        # Підсумок
        print("\nПІДСУМОК:")
        print(f"Навчено моделей: {len(self.models)} типів")
        print(f"Горизонти прогнозування: {self.horizons}")
        
        # Найкраща модель overall
        best_overall = self.evaluation_results.loc[
            self.evaluation_results['direction_accuracy'].idxmax()
        ]
        print(f"\nНайкраща модель загалом: {best_overall['model']} "
              f"(горизонт {best_overall['horizon']}h, "
              f"точність {best_overall['direction_accuracy']:.2%})")
        
        return self.evaluation_results


def main():
    forecaster = MultiModelForecasting()
    results = forecaster.run_full_pipeline()
    
    print("\n✅ Мульти-модельне прогнозування завершено!")
    print(f"Середня точність всіх моделей: {results['direction_accuracy'].mean():.2%}")


if __name__ == "__main__":
    main()