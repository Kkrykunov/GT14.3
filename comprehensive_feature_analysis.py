#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексний аналіз впливу КОЖНОЇ з 233 фічей
GT14 v14.2 - WhaleTracker

Цей модуль виконує систематичний аналіз всіх 233 універсальних фічей
для розуміння їх цінності та впливу на прогнозування btc_price
"""

import pandas as pd
import numpy as np
import mysql.connector
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFeatureAnalyzer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        self.features_df = None
        self.target = 'btc_price'
        self.analysis_results = {}
        
    def load_features_from_db(self):
        """Завантаження всіх 233 фічей з БД"""
        print("=== Завантаження даних з БД ===")
        
        conn = mysql.connector.connect(**self.db_config)
        
        # Завантажуємо універсальні фічі
        query = """
        SELECT DISTINCT feature_name 
        FROM universal_features 
        ORDER BY feature_name
        """
        cursor = conn.cursor()
        cursor.execute(query)
        feature_names = [row[0] for row in cursor.fetchall()]
        
        print(f"Знайдено {len(feature_names)} унікальних фічей")
        
        # Завантажуємо дані у форматі timestamp -> features
        query = """
        SELECT timestamp, feature_name, feature_value
        FROM universal_features
        ORDER BY timestamp, feature_name
        """
        
        cursor.execute(query)
        data = cursor.fetchall()
        
        # Перетворюємо в DataFrame
        df_long = pd.DataFrame(data, columns=['timestamp', 'feature_name', 'feature_value'])
        self.features_df = df_long.pivot(index='timestamp', columns='feature_name', values='feature_value')
        
        # Завантажуємо btc_price
        query = """
        SELECT timestamp, btc_price
        FROM whale_hourly_complete
        ORDER BY timestamp
        """
        cursor.execute(query)
        price_data = cursor.fetchall()
        
        price_df = pd.DataFrame(price_data, columns=['timestamp', 'btc_price'])
        price_df.set_index('timestamp', inplace=True)
        
        # Об'єднуємо
        self.features_df = self.features_df.merge(price_df, left_index=True, right_index=True)
        
        conn.close()
        
        print(f"Завантажено дані: {self.features_df.shape[0]} записів, {self.features_df.shape[1]} колонок")
        return self.features_df
    
    def etap1_individual_analysis(self):
        """Етап 1: Індивідуальний аналіз кожної фічі"""
        print("\n=== ЕТАП 1: Індивідуальний аналіз кожної фічі ===")
        
        results = []
        feature_cols = [col for col in self.features_df.columns if col != self.target]
        
        for i, feature in enumerate(feature_cols):
            if i % 20 == 0:
                print(f"Обробка фічі {i+1}/{len(feature_cols)}...")
            
            # Видаляємо NaN для аналізу
            data = self.features_df[[feature, self.target]].dropna()
            
            if len(data) < 10:
                continue
            
            X = data[feature].astype(float).values
            y = data[self.target].astype(float).values
            
            # Кореляція Пірсона
            pearson_corr, pearson_p = pearsonr(X, y)
            
            # Кореляція Спірмена
            spearman_corr, spearman_p = spearmanr(X, y)
            
            # Mutual Information
            mi_score = mutual_info_regression(X.reshape(-1, 1), y, random_state=42)[0]
            
            # F-статистика
            f_stat, f_p = f_regression(X.reshape(-1, 1), y)
            
            # Часова стабільність кореляції (rolling correlation)
            rolling_corr = pd.Series(X).rolling(window=100).corr(pd.Series(y))
            corr_stability = rolling_corr.std()
            
            results.append({
                'feature': feature,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'mutual_info': mi_score,
                'f_statistic': f_stat[0],
                'f_p_value': f_p[0],
                'corr_stability': corr_stability,
                'non_null_count': len(data),
                'null_percentage': (len(self.features_df) - len(data)) / len(self.features_df) * 100
            })
        
        self.etap1_results = pd.DataFrame(results)
        self.etap1_results = self.etap1_results.sort_values('mutual_info', ascending=False)
        
        # Зберігаємо топ-50 фічей
        print("\nТоп-20 фічей за Mutual Information:")
        print(self.etap1_results[['feature', 'mutual_info', 'pearson_corr', 'spearman_corr']].head(20))
        
        self.analysis_results['etap1'] = self.etap1_results
        return self.etap1_results
    
    def etap2_group_analysis(self):
        """Етап 2: Аналіз груп фічей"""
        print("\n=== ЕТАП 2: Аналіз груп фічей ===")
        
        # Визначаємо групи фічей
        groups = {
            'temporal': lambda x: any(s in x for s in ['hour', 'day', 'month', 'weekend', 'time']),
            'lag': lambda x: 'lag' in x,
            'rolling': lambda x: 'rolling' in x,
            'technical': lambda x: any(s in x for s in ['rsi', 'macd', 'bb_', 'stoch', 'williams']),
            'whale': lambda x: any(s in x for s in ['whale', 'flow', 'tx']),
            'volatility': lambda x: any(s in x for s in ['vol', 'change']),
            'interaction': lambda x: 'interaction' in x,
            'transform': lambda x: any(s in x for s in ['log', 'sqrt', 'zscore'])
        }
        
        group_results = []
        
        for group_name, group_func in groups.items():
            # Фільтруємо фічі для групи
            group_features = [f for f in self.etap1_results['feature'] if group_func(f)]
            
            if not group_features:
                continue
            
            # Статистика по групі
            group_data = self.etap1_results[self.etap1_results['feature'].isin(group_features)]
            
            result = {
                'group': group_name,
                'feature_count': len(group_features),
                'avg_mutual_info': group_data['mutual_info'].mean(),
                'max_mutual_info': group_data['mutual_info'].max(),
                'avg_pearson': group_data['pearson_corr'].abs().mean(),
                'max_pearson': group_data['pearson_corr'].abs().max(),
                'top_feature': group_data.iloc[0]['feature'] if len(group_data) > 0 else None,
                'features': group_features[:5]  # Топ-5 фічей групи
            }
            
            group_results.append(result)
        
        self.etap2_results = pd.DataFrame(group_results)
        self.etap2_results = self.etap2_results.sort_values('avg_mutual_info', ascending=False)
        
        print("\nРезультати аналізу груп:")
        print(self.etap2_results[['group', 'feature_count', 'avg_mutual_info', 'max_pearson']])
        
        self.analysis_results['etap2'] = self.etap2_results
        return self.etap2_results
    
    def etap3_multivariate_analysis(self):
        """Етап 3: Мультиваріантний аналіз"""
        print("\n=== ЕТАП 3: Мультиваріантний аналіз ===")
        
        # Беремо топ-50 фічей для детального аналізу
        top_features = self.etap1_results.head(50)['feature'].tolist()
        
        # Підготовка даних
        data = self.features_df[top_features + [self.target]].dropna()
        X = data[top_features]
        y = data[self.target]
        
        # 1. VIF аналіз для мультиколінеарності
        print("\n1. Аналіз мультиколінеарності (VIF)...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        vif_data = []
        for i, col in enumerate(X_scaled.columns):
            try:
                vif = variance_inflation_factor(X_scaled.values, i)
                vif_data.append({'feature': col, 'VIF': vif})
            except:
                vif_data.append({'feature': col, 'VIF': np.nan})
        
        vif_df = pd.DataFrame(vif_data)
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        print(f"\nФічі з високою мультиколінеарністю (VIF > 10):")
        print(vif_df[vif_df['VIF'] > 10].head(10))
        
        # 2. PCA для зменшення розмірності
        print("\n2. PCA аналіз...")
        pca = PCA(n_components=20)
        pca_result = pca.fit_transform(X_scaled)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"\nПояснена дисперсія перших 20 компонент: {cumulative_variance[-1]:.2%}")
        print(f"Компоненти для 95% дисперсії: {np.argmax(cumulative_variance >= 0.95) + 1}")
        
        # 3. Кластеризація фічей
        print("\n3. Кластеризація фічей...")
        # Кореляційна матриця між фічами
        feature_corr = X.corr()
        
        # Кластеризація на основі кореляції
        kmeans = KMeans(n_clusters=8, random_state=42)
        feature_clusters = kmeans.fit_predict(feature_corr)
        
        cluster_df = pd.DataFrame({
            'feature': top_features,
            'cluster': feature_clusters
        })
        
        print("\nРозподіл фічей по кластерах:")
        print(cluster_df['cluster'].value_counts().sort_index())
        
        self.analysis_results['etap3'] = {
            'vif_analysis': vif_df,
            'pca_variance': cumulative_variance,
            'feature_clusters': cluster_df
        }
        
        return self.analysis_results['etap3']
    
    def etap4_strategy_testing(self):
        """Етап 4: Тестування стратегій"""
        print("\n=== ЕТАП 4: Тестування стратегій ===")
        
        strategies = {}
        
        # Підготовка даних
        all_features = [col for col in self.features_df.columns if col != self.target]
        
        # Стратегія 1: Всі 233 фічі з LASSO
        print("\n1. Стратегія: Всі фічі з LASSO...")
        data = self.features_df.dropna()
        X = data[all_features]
        y = data[self.target]
        
        # Train-test split (80-20)
        split_point = int(len(data) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # LASSO
        lasso = Lasso(alpha=1.0, random_state=42)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_test)
        lasso_mape = mean_absolute_percentage_error(y_test, lasso_pred)
        
        # Відбір фічей з ненульовими коефіцієнтами
        lasso_features = [f for f, c in zip(all_features, lasso.coef_) if c != 0]
        
        strategies['all_features_lasso'] = {
            'feature_count': len(lasso_features),
            'mape': lasso_mape,
            'selected_features': lasso_features[:20]
        }
        
        # Стратегія 2: Топ-50 за importance
        print("\n2. Стратегія: Топ-50 фічей...")
        top50_features = self.etap1_results.head(50)['feature'].tolist()
        
        data_top50 = self.features_df[top50_features + [self.target]].dropna()
        X_top50 = data_top50[top50_features]
        y_top50 = data_top50[self.target]
        
        split_point = int(len(data_top50) * 0.8)
        X_train, X_test = X_top50[:split_point], X_top50[split_point:]
        y_train, y_test = y_top50[:split_point], y_top50[split_point:]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
        
        strategies['top50_rf'] = {
            'feature_count': 50,
            'mape': rf_mape,
            'feature_importance': dict(zip(top50_features[:10], rf.feature_importances_[:10]))
        }
        
        # Стратегія 3: Оптимальні комбінації груп
        print("\n3. Стратегія: Найкращі фічі з кожної групи...")
        best_group_features = []
        
        for _, row in self.etap2_results.iterrows():
            group_name = row['group']
            # Беремо топ-3 фічі з кожної групи
            group_features = self.etap1_results[
                self.etap1_results['feature'].isin(row['features'])
            ].head(3)['feature'].tolist()
            best_group_features.extend(group_features)
        
        data_groups = self.features_df[best_group_features + [self.target]].dropna()
        X_groups = data_groups[best_group_features]
        y_groups = data_groups[self.target]
        
        split_point = int(len(data_groups) * 0.8)
        X_train, X_test = X_groups[:split_point], X_groups[split_point:]
        y_train, y_test = y_groups[:split_point], y_groups[split_point:]
        
        rf_groups = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_groups.fit(X_train, y_train)
        groups_pred = rf_groups.predict(X_test)
        groups_mape = mean_absolute_percentage_error(y_test, groups_pred)
        
        strategies['best_from_groups'] = {
            'feature_count': len(best_group_features),
            'mape': groups_mape,
            'features_by_group': best_group_features[:20]
        }
        
        # Підсумок
        print("\n=== Підсумок стратегій ===")
        for name, results in strategies.items():
            print(f"\n{name}:")
            print(f"  Кількість фічей: {results['feature_count']}")
            print(f"  MAPE: {results['mape']:.2%}")
        
        self.analysis_results['etap4'] = strategies
        return strategies
    
    def etap5_documentation(self):
        """Етап 5: Документація результатів"""
        print("\n=== ЕТАП 5: Документація результатів ===")
        
        # 1. Зберігаємо таблицю всіх фічей з метриками
        self.etap1_results.to_csv('feature_analysis_full_233.csv', index=False)
        print("✓ Збережено: feature_analysis_full_233.csv")
        
        # 2. Зберігаємо топ-50 фічей
        self.etap1_results.head(50).to_csv('feature_analysis_top50.csv', index=False)
        print("✓ Збережено: feature_analysis_top50.csv")
        
        # 3. Зберігаємо аналіз груп
        self.etap2_results.to_csv('feature_group_analysis.csv', index=False)
        print("✓ Збережено: feature_group_analysis.csv")
        
        # 4. Візуалізація importance всіх 233 фічей
        plt.figure(figsize=(20, 10))
        
        # Топ-50 фічей
        top50 = self.etap1_results.head(50)
        
        plt.subplot(2, 1, 1)
        plt.bar(range(len(top50)), top50['mutual_info'])
        plt.title('Топ-50 фічей за Mutual Information Score')
        plt.xlabel('Фіча (ранг)')
        plt.ylabel('Mutual Information')
        plt.xticks([])
        
        # Розподіл по групах
        plt.subplot(2, 1, 2)
        self.etap2_results.plot(x='group', y='avg_mutual_info', kind='bar')
        plt.title('Середня важливість фічей по групах')
        plt.xlabel('Група фічей')
        plt.ylabel('Середній Mutual Information Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ Збережено: feature_importance_visualization.png")
        plt.close()
        
        # 5. Створюємо підсумковий звіт
        report = f"""
# Комплексний аналіз 233 універсальних фічей
## GT14 v14.2 - WhaleTracker

### Підсумок аналізу

**Всього проаналізовано фічей:** {len(self.etap1_results)}

### Топ-10 найважливіших фічей:
{self.etap1_results[['feature', 'mutual_info', 'pearson_corr']].head(10).to_string()}

### Аналіз груп фічей:
{self.etap2_results[['group', 'feature_count', 'avg_mutual_info']].to_string()}

### Результати тестування стратегій:
"""
        
        for name, results in self.analysis_results['etap4'].items():
            report += f"\n**{name}:**\n"
            report += f"- Кількість фічей: {results['feature_count']}\n"
            report += f"- MAPE: {results['mape']:.2%}\n"
        
        report += "\n### Рекомендації:\n"
        report += "1. Найкраща стратегія за MAPE: " + min(self.analysis_results['etap4'].items(), key=lambda x: x[1]['mape'])[0]
        report += "\n2. Рекомендована кількість фічей: 30-50 найважливіших"
        report += "\n3. Групи з найвищим впливом: " + ", ".join(self.etap2_results.head(3)['group'].tolist())
        
        with open('feature_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        print("✓ Збережено: feature_analysis_report.md")
        
        return report
    
    def run_full_analysis(self):
        """Запуск повного аналізу всіх етапів"""
        print("=== ПОЧАТОК КОМПЛЕКСНОГО АНАЛІЗУ 233 ФІЧЕЙ ===\n")
        
        # Завантаження даних
        self.load_features_from_db()
        
        # Етап 1
        self.etap1_individual_analysis()
        
        # Етап 2
        self.etap2_group_analysis()
        
        # Етап 3
        self.etap3_multivariate_analysis()
        
        # Етап 4
        self.etap4_strategy_testing()
        
        # Етап 5
        self.etap5_documentation()
        
        print("\n=== АНАЛІЗ ЗАВЕРШЕНО ===")
        print("Всі результати збережено у файли.")
        
        return self.analysis_results


def main():
    analyzer = ComprehensiveFeatureAnalyzer()
    results = analyzer.run_full_analysis()
    
    # Оновлюємо метадані фічей в БД з importance scores
    print("\nОновлення importance scores в БД...")
    conn = mysql.connector.connect(**analyzer.db_config)
    cursor = conn.cursor()
    
    for _, row in analyzer.etap1_results.iterrows():
        cursor.execute("""
        UPDATE feature_metadata 
        SET importance_score = %s
        WHERE feature_name = %s
        """, (float(row['mutual_info']), row['feature']))
    
    conn.commit()
    conn.close()
    print("✓ Importance scores оновлено в БД")


if __name__ == "__main__":
    main()