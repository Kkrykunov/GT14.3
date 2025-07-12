#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система динамічних фічей для GT14 v14.2
Автоматично створює нові фічі та відбирає найкращі
"""

import pandas as pd
import numpy as np
import mysql.connector
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class DynamicFeatureSystem:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        self.base_features = []
        self.generated_features = []
        self.selected_features = []
        self.feature_importance = {}
        
    def load_base_data(self):
        """Завантаження базових даних"""
        print("=== Завантаження базових даних ===")
        
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Завантажуємо базові колонки
        query = """
        SELECT timestamp, whale_volume_usd, whale_activity, 
               exchange_inflow, exchange_outflow, net_flow, 
               btc_price, fear_greed_index
        FROM whale_hourly_complete
        ORDER BY timestamp
        """
        
        cursor.execute(query)
        data = cursor.fetchall()
        
        columns = ['timestamp', 'whale_volume_usd', 'whale_activity', 
                  'exchange_inflow', 'exchange_outflow', 'net_flow', 
                  'btc_price', 'fear_greed_index']
        
        self.df = pd.DataFrame(data, columns=columns)
        self.df.set_index('timestamp', inplace=True)
        
        self.base_features = [col for col in columns if col not in ['timestamp', 'btc_price']]
        
        conn.close()
        
        print(f"Завантажено {len(self.df)} записів")
        print(f"Базові фічі: {self.base_features}")
        
        return self.df
    
    def generate_polynomial_features(self, degree=2):
        """Генерація поліноміальних фічей"""
        print(f"\n=== Генерація поліноміальних фічей (degree={degree}) ===")
        
        # Вибираємо числові фічі
        numeric_features = ['whale_volume_usd', 'whale_activity', 'net_flow']
        X = self.df[numeric_features].fillna(0)
        
        # Створюємо поліноміальні фічі
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(X)
        
        # Отримуємо назви фічей
        feature_names = poly.get_feature_names_out(numeric_features)
        
        # Додаємо до DataFrame
        for i, name in enumerate(feature_names):
            if name not in self.df.columns:
                self.df[f'poly_{name}'] = poly_features[:, i]
                self.generated_features.append(f'poly_{name}')
        
        print(f"Згенеровано {len(feature_names)} поліноміальних фічей")
    
    def generate_interaction_features(self):
        """Генерація фічей взаємодії"""
        print("\n=== Генерація фічей взаємодії ===")
        
        # Визначаємо пари для взаємодії
        interaction_pairs = [
            ('whale_volume_usd', 'fear_greed_index'),
            ('net_flow', 'whale_activity'),
            ('exchange_inflow', 'exchange_outflow'),
            ('whale_volume_usd', 'btc_price'),
            ('whale_activity', 'fear_greed_index')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in self.df.columns and feat2 in self.df.columns:
                # Множення
                interaction_name = f'{feat1}_x_{feat2}'
                self.df[interaction_name] = self.df[feat1] * self.df[feat2]
                self.generated_features.append(interaction_name)
                
                # Відношення
                ratio_name = f'{feat1}_div_{feat2}'
                self.df[ratio_name] = self.df[feat1] / (self.df[feat2] + 1e-8)
                self.generated_features.append(ratio_name)
        
        print(f"Згенеровано {len(interaction_pairs) * 2} фічей взаємодії")
    
    def generate_domain_specific_features(self):
        """Генерація доменно-специфічних фічей"""
        print("\n=== Генерація доменно-специфічних фічей ===")
        
        # Whale pressure index
        self.df['whale_pressure_index'] = (
            self.df['whale_volume_usd'] * self.df['whale_activity'] / 
            self.df['btc_price']
        )
        self.generated_features.append('whale_pressure_index')
        
        # Market momentum
        self.df['market_momentum'] = (
            self.df['net_flow'].rolling(6).mean() * 
            self.df['btc_price'].pct_change(6)
        )
        self.generated_features.append('market_momentum')
        
        # Fear-adjusted volume
        self.df['fear_adjusted_volume'] = (
            self.df['whale_volume_usd'] * 
            (100 - self.df['fear_greed_index']) / 100
        )
        self.generated_features.append('fear_adjusted_volume')
        
        # Exchange flow imbalance momentum
        self.df['flow_imbalance_momentum'] = (
            self.df['net_flow'].rolling(12).mean() - 
            self.df['net_flow'].rolling(48).mean()
        )
        self.generated_features.append('flow_imbalance_momentum')
        
        # Whale activity acceleration
        self.df['whale_activity_acceleration'] = (
            self.df['whale_activity'].diff().rolling(6).mean()
        )
        self.generated_features.append('whale_activity_acceleration')
        
        print(f"Згенеровано 5 доменно-специфічних фічей")
    
    def automatic_feature_detection(self):
        """Автоматичне виявлення нових патернів фічей"""
        print("\n=== Автоматичне виявлення фічей ===")
        
        # Виявлення трендових змін
        for feature in self.base_features:
            if feature in self.df.columns:
                # Зміна тренду
                self.df[f'{feature}_trend_change'] = (
                    self.df[feature].rolling(24).mean() - 
                    self.df[feature].rolling(168).mean()
                )
                self.generated_features.append(f'{feature}_trend_change')
                
                # Volatility ratio
                self.df[f'{feature}_vol_ratio'] = (
                    self.df[feature].rolling(24).std() / 
                    (self.df[feature].rolling(168).std() + 1e-8)
                )
                self.generated_features.append(f'{feature}_vol_ratio')
        
        print(f"Автоматично виявлено {len(self.base_features) * 2} нових фічей")
    
    def generate_all_features(self):
        """Генерація всіх типів фічей"""
        print("\n=== ГЕНЕРАЦІЯ ВСІХ ДИНАМІЧНИХ ФІЧЕЙ ===")
        
        # Генеруємо всі типи фічей
        self.generate_polynomial_features(degree=2)
        self.generate_interaction_features()
        self.generate_domain_specific_features()
        self.automatic_feature_detection()
        
        print(f"\nВсього згенеровано динамічних фічей: {len(self.generated_features)}")
        
        return self.df
    
    def select_features_mutual_info(self, k=50):
        """Відбір фічей за mutual information"""
        print(f"\n=== Відбір топ-{k} фічей за Mutual Information ===")
        
        # Підготовка даних
        feature_cols = self.base_features + self.generated_features
        X = self.df[feature_cols].fillna(0)
        y = self.df['btc_price'].fillna(method='ffill')
        
        # Видаляємо рядки з NaN в цільовій змінній
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Створюємо DataFrame з результатами
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Відбираємо топ-k
        top_features = mi_df.head(k)['feature'].tolist()
        self.selected_features.extend(top_features)
        
        print(f"Топ-10 фічей:")
        print(mi_df.head(10))
        
        return mi_df
    
    def select_features_rfe(self, n_features=30):
        """Відбір фічей за допомогою RFE"""
        print(f"\n=== Відбір {n_features} фічей за допомогою RFE ===")
        
        # Підготовка даних
        feature_cols = self.base_features + self.generated_features
        X = self.df[feature_cols].fillna(0)
        y = self.df['btc_price'].fillna(method='ffill')
        
        # Видаляємо рядки з NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Беремо підвибірку для швидкості
        sample_size = min(1000, len(X))
        X_sample = X.iloc[:sample_size]
        y_sample = y.iloc[:sample_size]
        
        # RFE з RandomForest
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X_sample, y_sample)
        
        # Відібрані фічі
        rfe_features = [f for f, s in zip(feature_cols, rfe.support_) if s]
        self.selected_features.extend(rfe_features)
        
        print(f"Відібрано {len(rfe_features)} фічей через RFE")
        
        return rfe_features
    
    def feature_importance_analysis(self):
        """Аналіз важливості фічей"""
        print("\n=== Аналіз важливості фічей ===")
        
        # Використовуємо унікальні відібрані фічі
        unique_features = list(set(self.selected_features))
        
        if not unique_features:
            unique_features = self.base_features + self.generated_features
        
        # Підготовка даних
        X = self.df[unique_features].fillna(0)
        y = self.df['btc_price'].fillna(method='ffill')
        
        # Видаляємо рядки з NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # RandomForest для importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Зберігаємо importance
        importance_df = pd.DataFrame({
            'feature': unique_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = dict(zip(importance_df['feature'], 
                                         importance_df['importance']))
        
        print("\nТоп-15 найважливіших фічей:")
        print(importance_df.head(15))
        
        return importance_df
    
    def save_dynamic_features(self):
        """Збереження динамічних фічей в БД"""
        print("\n=== Збереження динамічних фічей ===")
        
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Створюємо таблицю для динамічних фічей
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dynamic_features (
            timestamp DATETIME,
            feature_name VARCHAR(100),
            feature_value DOUBLE,
            feature_importance DECIMAL(10, 8),
            generation_method VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (timestamp, feature_name)
        )
        """)
        
        # Підготовка даних для вставки
        insert_data = []
        
        for feature in self.generated_features:
            if feature in self.df.columns:
                importance = self.feature_importance.get(feature, 0.0)
                
                # Визначаємо метод генерації
                if 'poly_' in feature:
                    method = 'polynomial'
                elif '_x_' in feature or '_div_' in feature:
                    method = 'interaction'
                elif any(x in feature for x in ['trend_change', 'vol_ratio']):
                    method = 'automatic_detection'
                else:
                    method = 'domain_specific'
                
                for timestamp, value in self.df[feature].items():
                    if pd.notna(value):
                        insert_data.append((
                            timestamp,
                            feature,
                            float(value),
                            float(importance),
                            method
                        ))
        
        # Вставка батчами
        if insert_data:
            batch_size = 1000
            for i in range(0, len(insert_data), batch_size):
                batch = insert_data[i:i + batch_size]
                cursor.executemany("""
                INSERT INTO dynamic_features 
                (timestamp, feature_name, feature_value, feature_importance, generation_method)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                feature_value = VALUES(feature_value),
                feature_importance = VALUES(feature_importance)
                """, batch)
                print(f"Збережено батч {i//batch_size + 1}: {len(batch)} записів")
        
        conn.commit()
        conn.close()
        
        print(f"\nВсього збережено {len(insert_data)} записів динамічних фічей")
    
    def get_optimal_feature_set(self):
        """Повертає оптимальний набір фічей"""
        # Об'єднуємо всі методи відбору
        all_selected = list(set(self.selected_features))
        
        # Сортуємо за importance
        sorted_features = sorted(
            all_selected, 
            key=lambda x: self.feature_importance.get(x, 0), 
            reverse=True
        )
        
        return sorted_features[:50]  # Топ-50 фічей
    
    def run_pipeline(self):
        """Запуск повного pipeline"""
        print("=== ЗАПУСК СИСТЕМИ ДИНАМІЧНИХ ФІЧЕЙ ===\n")
        
        # 1. Завантаження даних
        self.load_base_data()
        
        # 2. Генерація фічей
        self.generate_polynomial_features(degree=2)
        self.generate_interaction_features()
        self.generate_domain_specific_features()
        self.automatic_feature_detection()
        
        print(f"\nВсього згенеровано нових фічей: {len(self.generated_features)}")
        
        # 3. Відбір фічей
        self.select_features_mutual_info(k=50)
        self.select_features_rfe(n_features=30)
        
        # 4. Аналіз важливості
        self.feature_importance_analysis()
        
        # 5. Збереження
        self.save_dynamic_features()
        
        # 6. Фінальний набір
        optimal_features = self.get_optimal_feature_set()
        
        print(f"\n=== ОПТИМАЛЬНИЙ НАБІР ФІЧЕЙ ===")
        print(f"Кількість: {len(optimal_features)}")
        print(f"Топ-10: {optimal_features[:10]}")
        
        # Збереження результатів
        results_df = pd.DataFrame({
            'feature': optimal_features,
            'importance': [self.feature_importance.get(f, 0) for f in optimal_features]
        })
        results_df.to_csv('dynamic_features_optimal_set.csv', index=False)
        
        return optimal_features


def main():
    system = DynamicFeatureSystem()
    optimal_features = system.run_pipeline()
    
    print("\n✅ Система динамічних фічей успішно виконана!")
    print(f"Оптимальний набір містить {len(optimal_features)} фічей")
    print("Результати збережено в dynamic_features_optimal_set.csv")


if __name__ == "__main__":
    main()