#!/usr/bin/env python3
"""
Інтерактивна кластеризація з вибором параметрів користувачем
"""

import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class InteractiveClustering:
    def __init__(self):
        self.config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        
    def run_interactive_clustering(self):
        print(" ІНТЕРАКТИВНА КЛАСТЕРИЗАЦІЯ WHALE TRACKER")
        print("=" * 60)
        
        # 1. Завантаження даних
        df = self.load_data()
        
        # 2. Вибір змінних для кластеризації
        features = self.select_features(df)
        
        # 3. Вибір методу кластеризації
        clustering_method = self.select_clustering_method()
        
        # 4. Налаштування параметрів
        params = self.configure_parameters(clustering_method)
        
        # 5. Виконання кластеризації
        results = self.perform_clustering(df, features, clustering_method, params)
        
        # 6. Збереження результатів
        self.save_results(df, results, features, clustering_method, params)
        
        return results
    
    def load_data(self):
        """Завантаження даних з MySQL"""
        print("\n ЗАВАНТАЖЕННЯ ДАНИХ...")
        
        conn = mysql.connector.connect(**self.config)
        
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
        
        df = pd.read_sql(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        conn.close()
        
        print(f" Завантажено {len(df)} записів")
        return df
    
    def select_features(self, df):
        """Вибір змінних для кластеризації"""
        print("\n ВИБІР ЗМІННИХ ДЛЯ КЛАСТЕРИЗАЦІЇ")
        print("-" * 40)
        
        available_features = [
            'whale_volume_usd', 'whale_activity', 'exchange_inflow', 'exchange_outflow', 
            'net_flow', 'btc_price', 'fear_greed_index', 'SP500', 'VIX', 'GOLD', 'NASDAQ', 'OIL_WTI'
        ]
        
        print("Доступні змінні:")
        for i, feature in enumerate(available_features, 1):
            non_null_count = df[feature].notna().sum() if feature in df.columns else 0
            print(f"  {i:2}. {feature} (не-null: {non_null_count:,})")
        
        print(\"\\nВиберіть змінні для кластеризації:\")\n        print(\"1. Стандартний набір (whale_volume_usd, net_flow, whale_activity, exchange_inflow, exchange_outflow, fear_greed_index)\")\n        print(\"2. Розширений з традиційними ринками\")\n        print(\"3. Тільки whale метрики\")\n        print(\"4. Вручну\")\n        \n        choice = input(\"\\nВаш вибір (1-4): \").strip()\n        \n        if choice == '1':\n            selected_features = ['whale_volume_usd', 'net_flow', 'whale_activity', \n                               'exchange_inflow', 'exchange_outflow', 'fear_greed_index']\n        elif choice == '2':\n            selected_features = ['whale_volume_usd', 'net_flow', 'whale_activity', \n                               'exchange_inflow', 'exchange_outflow', 'fear_greed_index',\n                               'SP500', 'VIX', 'GOLD']\n        elif choice == '3':\n            selected_features = ['whale_volume_usd', 'whale_activity', 'exchange_inflow', 'exchange_outflow']\n        elif choice == '4':\n            selected_features = []\n            print(\"\\nВведіть номери змінних через кому (наприклад: 1,3,5):\")\n            indices = input().strip().split(',')\n            for idx in indices:\n                try:\n                    feature_idx = int(idx.strip()) - 1\n                    if 0 <= feature_idx < len(available_features):\n                        selected_features.append(available_features[feature_idx])\n                except:\n                    pass\n        else:\n            # За замовчуванням\n            selected_features = ['whale_volume_usd', 'net_flow', 'whale_activity', \n                               'exchange_inflow', 'exchange_outflow', 'fear_greed_index']\n        \n        # Фільтруємо доступні\n        selected_features = [f for f in selected_features if f in df.columns and df[f].notna().sum() > 100]\n        \n        print(f\"\\n Обрані змінні: {selected_features}\")\n        return selected_features\n    \n    def select_clustering_method(self):\n        \"\"\"Вибір методу кластеризації\"\"\"\n        print(\"\\n ВИБІР МЕТОДУ КЛАСТЕРИЗАЦІЇ\")\n        print(\"-\" * 40)\n        \n        methods = {\n            '1': 'KMeans',\n            '2': 'DBSCAN', \n            '3': 'AgglomerativeClustering'\n        }\n        \n        print(\"Доступні методи:\")\n        print(\"1. K-Means (найпопулярніший, потребує вказати кількість кластерів)\")\n        print(\"2. DBSCAN (автоматично визначає кількість кластерів, працює з шумом)\")\n        print(\"3. Agglomerative (ієрархічна кластеризація)\")\n        \n        choice = input(\"\\nВаш вибір (1-3): \").strip()\n        method = methods.get(choice, 'KMeans')\n        \n        print(f\" Обраний метод: {method}\")\n        return method\n    \n    def configure_parameters(self, method):\n        \"\"\"Налаштування параметрів кластеризації\"\"\"\n        print(f\"\\n НАЛАШТУВАННЯ ПАРАМЕТРІВ ДЛЯ {method}\")\n        print(\"-\" * 40)\n        \n        params = {}\n        \n        if method == 'KMeans':\n            print(\"Параметри для K-Means:\")\n            \n            # Кількість кластерів\n            n_clusters = input(\"Кількість кластерів (за замовчуванням 7): \").strip()\n            params['n_clusters'] = int(n_clusters) if n_clusters.isdigit() else 7\n            \n            # Діапазон для автоматичного вибору\n            auto_select = input(\"Автоматично вибрати оптимальну кількість? (y/n): \").strip().lower()\n            if auto_select == 'y':\n                min_k = input(\"Мінімальна кількість кластерів (за замовчуванням 2): \").strip()\n                max_k = input(\"Максимальна кількість кластерів (за замовчуванням 15): \").strip()\n                params['auto_select'] = True\n                params['min_k'] = int(min_k) if min_k.isdigit() else 2\n                params['max_k'] = int(max_k) if max_k.isdigit() else 15\n            else:\n                params['auto_select'] = False\n                \n        elif method == 'DBSCAN':\n            print(\"Параметри для DBSCAN:\")\n            \n            eps = input(\"Параметр eps (відстань, за замовчуванням 0.5): \").strip()\n            params['eps'] = float(eps) if eps else 0.5\n            \n            min_samples = input(\"Мінімальна кількість точок в кластері (за замовчуванням 5): \").strip()\n            params['min_samples'] = int(min_samples) if min_samples.isdigit() else 5\n            \n        elif method == 'AgglomerativeClustering':\n            print(\"Параметри для Agglomerative Clustering:\")\n            \n            n_clusters = input(\"Кількість кластерів (за замовчуванням 7): \").strip()\n            params['n_clusters'] = int(n_clusters) if n_clusters.isdigit() else 7\n            \n            linkage = input(\"Метод зв'язування (ward/complete/average, за замовчуванням ward): \").strip()\n            params['linkage'] = linkage if linkage in ['ward', 'complete', 'average'] else 'ward'\n        \n        print(f\" Параметри: {params}\")\n        return params\n    \n    def perform_clustering(self, df, features, method, params):\n        \"\"\"Виконання кластеризації\"\"\"\n        print(f\"\\n ВИКОНАННЯ КЛАСТЕРИЗАЦІЇ {method}\")\n        print(\"-\" * 40)\n        \n        # Підготовка даних\n        X = df[features].dropna()\n        \n        # Нормалізація\n        scaler = StandardScaler()\n        X_scaled = scaler.fit_transform(X)\n        \n        results = {\n            'method': method,\n            'features': features,\n            'params': params,\n            'data_points': len(X),\n            'scaler': scaler\n        }\n        \n        if method == 'KMeans':\n            if params.get('auto_select', False):\n                # Автоматичний вибір оптимальної кількості\n                best_k, metrics = self.find_optimal_k(X_scaled, params['min_k'], params['max_k'])\n                params['n_clusters'] = best_k\n                results['optimization_metrics'] = metrics\n                print(f\" Оптимальна кількість кластерів: {best_k}\")\n            \n            clusterer = KMeans(n_clusters=params['n_clusters'], random_state=42)\n            labels = clusterer.fit_predict(X_scaled)\n            results['cluster_centers'] = clusterer.cluster_centers_\n            \n        elif method == 'DBSCAN':\n            clusterer = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])\n            labels = clusterer.fit_predict(X_scaled)\n            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)\n            print(f\" Знайдено кластерів: {n_clusters}\")\n            \n        elif method == 'AgglomerativeClustering':\n            clusterer = AgglomerativeClustering(\n                n_clusters=params['n_clusters'], \n                linkage=params['linkage']\n            )\n            labels = clusterer.fit_predict(X_scaled)\n        \n        # Обчислення метрик\n        if len(set(labels)) > 1:\n            results['silhouette'] = silhouette_score(X_scaled, labels)\n            results['davies_bouldin'] = davies_bouldin_score(X_scaled, labels)\n            results['calinski_harabasz'] = calinski_harabasz_score(X_scaled, labels)\n        else:\n            results['silhouette'] = 0\n            results['davies_bouldin'] = float('inf')\n            results['calinski_harabasz'] = 0\n        \n        results['labels'] = labels\n        results['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)\n        results['X_original'] = X\n        results['X_scaled'] = X_scaled\n        \n        print(f\" Кластеризація завершена:\")\n        print(f\"   Кластерів: {results['n_clusters']}\")\n        print(f\"   Silhouette: {results['silhouette']:.3f}\")\n        print(f\"   Davies-Bouldin: {results['davies_bouldin']:.3f}\")\n        print(f\"   Calinski-Harabasz: {results['calinski_harabasz']:.1f}\")\n        \n        return results\n    \n    def find_optimal_k(self, X_scaled, min_k, max_k):\n        \"\"\"Знаходження оптимальної кількості кластерів\"\"\"\n        print(f\" Пошук оптимальної кількості кластерів ({min_k}-{max_k})...\")\n        \n        k_range = range(min_k, max_k + 1)\n        metrics = {\n            'k': [],\n            'silhouette': [],\n            'davies_bouldin': [],\n            'calinski_harabasz': [],\n            'inertia': []\n        }\n        \n        for k in k_range:\n            kmeans = KMeans(n_clusters=k, random_state=42)\n            labels = kmeans.fit_predict(X_scaled)\n            \n            metrics['k'].append(k)\n            metrics['silhouette'].append(silhouette_score(X_scaled, labels))\n            metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))\n            metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))\n            metrics['inertia'].append(kmeans.inertia_)\n            \n            print(f\"  K={k}: Silhouette={metrics['silhouette'][-1]:.3f}\")\n        \n        # Вибір оптимального K (найвищий silhouette score)\n        best_k_idx = np.argmax(metrics['silhouette'])\n        best_k = metrics['k'][best_k_idx]\n        \n        return best_k, metrics\n    \n    def save_results(self, df, results, features, method, params):\n        \"\"\"Збереження результатів кластеризації\"\"\"\n        print(f\"\\n ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ\")\n        print(\"-\" * 40)\n        \n        # Збереження в MySQL\n        conn = mysql.connector.connect(**self.config)\n        cursor = conn.cursor()\n        \n        # Створення таблиці для інтерактивних результатів\n        cursor.execute(\"\"\"\n        CREATE TABLE IF NOT EXISTS interactive_clustering_results (\n            id INT AUTO_INCREMENT PRIMARY KEY,\n            timestamp DATETIME,\n            cluster_id INT,\n            method VARCHAR(50),\n            features TEXT,\n            params TEXT,\n            silhouette_score DECIMAL(10,6),\n            davies_bouldin_score DECIMAL(10,6),\n            calinski_harabasz_score DECIMAL(15,2),\n            n_clusters INT,\n            data_points INT,\n            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n        )\n        \"\"\")\n        \n        # Очищення старих результатів\n        cursor.execute(\"DELETE FROM interactive_clustering_results\")\n        \n        # Вставка нових результатів\n        X_original = results['X_original']\n        labels = results['labels']\n        \n        insert_data = []\n        for idx, (timestamp, cluster_id) in enumerate(zip(X_original.index, labels)):\n            insert_data.append((\n                timestamp,\n                int(cluster_id),\n                method,\n                ','.join(features),\n                str(params),\n                float(results['silhouette']),\n                float(results['davies_bouldin']),\n                float(results['calinski_harabasz']),\n                results['n_clusters'],\n                results['data_points']\n            ))\n        \n        cursor.executemany(\"\"\"\n        INSERT INTO interactive_clustering_results \n        (timestamp, cluster_id, method, features, params, silhouette_score, \n         davies_bouldin_score, calinski_harabasz_score, n_clusters, data_points)\n        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n        \"\"\", insert_data)\n        \n        conn.commit()\n        conn.close()\n        \n        print(f\" Збережено {len(insert_data)} записів в interactive_clustering_results\")\n        \n        # Візуалізація\n        self.create_visualizations(results)\n        \n        return True\n    \n    def create_visualizations(self, results):\n        \"\"\"Створення візуалізацій\"\"\"\n        print(\"\\n СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ...\")\n        \n        X_scaled = results['X_scaled']\n        labels = results['labels']\n        features = results['features']\n        \n        # PCA візуалізація\n        pca = PCA(n_components=2)\n        X_pca = pca.fit_transform(X_scaled)\n        \n        plt.figure(figsize=(12, 8))\n        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)\n        plt.colorbar(scatter, label='Кластер')\n        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')\n        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')\n        plt.title(f'Інтерактивна кластеризація ({results[\"method\"]}) - {results[\"n_clusters\"]} кластерів')\n        plt.tight_layout()\n        plt.savefig('interactive_clustering_pca.png', dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        print(\" Візуалізації збережено: interactive_clustering_pca.png\")\n\ndef main():\n    clustering = InteractiveClustering()\n    results = clustering.run_interactive_clustering()\n    \n    print(\"\\n ІНТЕРАКТИВНА КЛАСТЕРИЗАЦІЯ ЗАВЕРШЕНА!\")\n    print(f\"Результат: {results['n_clusters']} кластерів з метриками:\")\n    print(f\"- Silhouette: {results['silhouette']:.3f}\")\n    print(f\"- Davies-Bouldin: {results['davies_bouldin']:.3f}\")\n    print(f\"- Calinski-Harabasz: {results['calinski_harabasz']:.1f}\")\n\nif __name__ == \"__main__\":\n    main()