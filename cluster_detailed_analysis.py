#!/usr/bin/env python3
"""
Детальний аналіз всіх 7 кластерів по всім метрикам
"""

import mysql.connector
import pandas as pd
import numpy as np

def analyze_all_clusters():
    print(" ДЕТАЛЬНИЙ АНАЛІЗ ВСІХ 7 КЛАСТЕРІВ")
    print("=" * 80)
    
    # Підключення до БД
    config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'whale_tracker_2024',
        'database': 'gt14_whaletracker'
    }
    
    conn = mysql.connector.connect(**config)
    
    # Завантажуємо дані кластерів
    cluster_query = """
    SELECT c.timestamp, c.cluster_id, c.whale_volume_usd, c.net_flow, 
           c.whale_activity, c.exchange_inflow, c.exchange_outflow, c.fear_greed_index,
           f.cluster_name
    FROM cluster_labels c
    JOIN cluster_features f ON c.cluster_id = f.cluster_id
    ORDER BY c.cluster_id, c.timestamp
    """
    
    df_clusters = pd.read_sql(cluster_query, conn)
    
    # Характеристики кластерів
    cluster_features_query = """
    SELECT * FROM cluster_features ORDER BY cluster_id
    """
    
    df_cluster_features = pd.read_sql(cluster_features_query, conn)
    conn.close()
    
    print(f" Загальна кількість записів: {len(df_clusters)}")
    print(f"  Кількість кластерів: {df_clusters['cluster_id'].nunique()}")
    print()
    
    # Детальний аналіз кожного кластера
    for cluster_id in sorted(df_clusters['cluster_id'].unique()):
        cluster_data = df_clusters[df_clusters['cluster_id'] == cluster_id]
        cluster_info = df_cluster_features[df_cluster_features['cluster_id'] == cluster_id].iloc[0]
        
        print(f" КЛАСТЕР {cluster_id}: {cluster_info['cluster_name']}")
        print("-" * 60)
        
        print(f" ОСНОВНІ МЕТРИКИ:")
        print(f"  Кількість записів: {len(cluster_data):,}")
        print(f"  Відсоток від загальних даних: {len(cluster_data)/len(df_clusters)*100:.1f}%")
        print()
        
        print(f" WHALE МЕТРИКИ:")
        print(f"  Середній Whale Volume: ${cluster_data['whale_volume_usd'].mean():,.0f}")
        print(f"  Медіана Whale Volume: ${cluster_data['whale_volume_usd'].median():,.0f}")
        print(f"  Максимальний Whale Volume: ${cluster_data['whale_volume_usd'].max():,.0f}")
        print(f"  Стандартне відхилення: ${cluster_data['whale_volume_usd'].std():,.0f}")
        print()
        
        print(f"  Середня активність китів: {cluster_data['whale_activity'].mean():.1f}")
        print(f"  Максимальна активність: {cluster_data['whale_activity'].max()}")
        print()
        
        print(f" FLOW МЕТРИКИ:")
        print(f"  Середній Net Flow: ${cluster_data['net_flow'].mean():,.0f}")
        print(f"  Медіана Net Flow: ${cluster_data['net_flow'].median():,.0f}")
        print(f"  Мін/Макс Net Flow: ${cluster_data['net_flow'].min():,.0f} / ${cluster_data['net_flow'].max():,.0f}")
        print()
        
        print(f"  Середній Exchange Inflow: ${cluster_data['exchange_inflow'].mean():,.0f}")
        print(f"  Середній Exchange Outflow: ${cluster_data['exchange_outflow'].mean():,.0f}")
        print()
        
        print(f" FEAR & GREED МЕТРИКИ:")
        print(f"  Середній Fear & Greed Index: {cluster_data['fear_greed_index'].mean():.1f}")
        print(f"  Медіана Fear & Greed: {cluster_data['fear_greed_index'].median():.1f}")
        print(f"  Мін/Макс Fear & Greed: {cluster_data['fear_greed_index'].min()} / {cluster_data['fear_greed_index'].max()}")
        print()
        
        # Характеристика настрою
        avg_fear_greed = cluster_data['fear_greed_index'].mean()
        if avg_fear_greed >= 75:
            sentiment = " КРАЙНЯ ЖАДІБНІСТЬ"
        elif avg_fear_greed >= 55:
            sentiment = " ЖАДІБНІСТЬ"
        elif avg_fear_greed >= 45:
            sentiment = " НЕЙТРАЛЬНИЙ"
        elif avg_fear_greed >= 25:
            sentiment = " СТРАХ"
        else:
            sentiment = " КРАЙНІЙ СТРАХ"
        
        print(f"  Загальний настрій кластера: {sentiment}")
        print()
        
        # Часові характеристики
        cluster_data['timestamp'] = pd.to_datetime(cluster_data['timestamp'])
        print(f" ЧАСОВІ ХАРАКТЕРИСТИКИ:")
        print(f"  Період: {cluster_data['timestamp'].min()} до {cluster_data['timestamp'].max()}")
        
        # Активність по годинах
        cluster_data['hour'] = cluster_data['timestamp'].dt.hour
        most_active_hour = cluster_data.groupby('hour')['whale_volume_usd'].sum().idxmax()
        print(f"  Найактивніша година: {most_active_hour}:00")
        
        # Активність по днях тижня
        cluster_data['dayofweek'] = cluster_data['timestamp'].dt.dayofweek
        days = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', "П'ятниця", 'Субота', 'Неділя']
        most_active_day = cluster_data.groupby('dayofweek')['whale_volume_usd'].sum().idxmax()
        print(f"  Найактивніший день: {days[most_active_day]}")
        print()
        
        # Кореляції в кластері
        numeric_cols = ['whale_volume_usd', 'net_flow', 'whale_activity', 
                       'exchange_inflow', 'exchange_outflow', 'fear_greed_index']
        
        cluster_corr = cluster_data[numeric_cols].corr()
        
        print(f" КЛЮЧОВІ КОРЕЛЯЦІЇ В КЛАСТЕРІ:")
        # Знаходимо найсильніші кореляції (крім діагоналі)
        correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                var1, var2 = numeric_cols[i], numeric_cols[j]
                corr_val = cluster_corr.iloc[i, j]
                if abs(corr_val) > 0.3:  # Показуємо тільки сильні кореляції
                    correlations.append((var1, var2, corr_val))
        
        # Сортуємо по силі кореляції
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for var1, var2, corr_val in correlations[:5]:  # Топ 5 кореляцій
            direction = "" if corr_val > 0 else ""
            print(f"  {direction} {var1} ↔ {var2}: {corr_val:.3f}")
        
        if not correlations:
            print("  Немає сильних кореляцій (>0.3)")
        
        print()
        
        # Статистичні характеристики
        print(f" СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ:")
        
        # Коефіцієнт варіації
        cv_whale = cluster_data['whale_volume_usd'].std() / cluster_data['whale_volume_usd'].mean()
        print(f"  Коефіцієнт варіації Whale Volume: {cv_whale:.2f}")
        
        # Асиметрія (skewness)
        skew_whale = cluster_data['whale_volume_usd'].skew()
        print(f"  Асиметрія Whale Volume: {skew_whale:.2f}")
        
        # Ексцес (kurtosis)
        kurt_whale = cluster_data['whale_volume_usd'].kurtosis()
        print(f"  Ексцес Whale Volume: {kurt_whale:.2f}")
        
        print()
        
        # Особливості кластера
        print(f" ОСОБЛИВОСТІ КЛАСТЕРА:")
        
        # Аналіз позицій
        positive_flow = (cluster_data['net_flow'] > 0).sum()
        negative_flow = (cluster_data['net_flow'] < 0).sum()
        neutral_flow = (cluster_data['net_flow'] == 0).sum()
        
        print(f"  Позитивний flow: {positive_flow} ({positive_flow/len(cluster_data)*100:.1f}%)")
        print(f"  Негативний flow: {negative_flow} ({negative_flow/len(cluster_data)*100:.1f}%)")
        print(f"  Нейтральний flow: {neutral_flow} ({neutral_flow/len(cluster_data)*100:.1f}%)")
        
        # Екстремальні значення
        top_5_whale = cluster_data.nlargest(5, 'whale_volume_usd')
        print(f"  Топ-5 whale транзакцій: ${top_5_whale['whale_volume_usd'].min():,.0f} - ${top_5_whale['whale_volume_usd'].max():,.0f}")
        
        print()
        print("=" * 80)
        print()

if __name__ == "__main__":
    analyze_all_clusters()