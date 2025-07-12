#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Importance Analysis для GT14 v14.2
Аналіз важливості відібраних фічей різними методами
"""

import pandas as pd
import numpy as np
import mysql.connector
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_importance_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Аналіз важливості фічей різними методами"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        self.importance_results = {}
        self.top_features = {}
        logger.info("Ініціалізація FeatureImportanceAnalyzer")
        
    def load_data_with_features(self):
        """Завантаження даних з усіма фічами"""
        logger.info("=== Завантаження даних з усіма фічами ===")
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Основний запит для даних з таблиці фічей
            query = """
            SELECT *
            FROM whale_features_basic
            WHERE btc_price > 0
            ORDER BY timestamp DESC
            LIMIT 5000
            """
            
            df = pd.read_sql(query, conn)
            logger.info(f"Завантажено {len(df)} записів")
            
            # Завантаження метаданих фічей
            metadata_query = """
            SELECT feature_name, feature_type, importance_score
            FROM feature_metadata
            WHERE importance_score > 0
            ORDER BY importance_score DESC
            """
            
            feature_metadata = pd.read_sql(metadata_query, conn)
            logger.info(f"Завантажено метадані для {len(feature_metadata)} фічей")
            
            conn.close()
            
            # Видаляємо нечислові колонки
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols]
            
            # Видаляємо колонки з NaN
            df_clean = df_numeric.dropna(axis=1)
            
            logger.info(f"Після очищення: {df_clean.shape[1]} фічей")
            
            return df_clean, feature_metadata
            
        except Exception as e:
            logger.error(f"Помилка завантаження даних: {str(e)}")
            raise
    
    def calculate_rf_importance(self, X, y, feature_names):
        """Розрахунок важливості через RandomForest"""
        logger.info("=== Розрахунок RandomForest importance ===")
        
        # Розділення даних
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Навчання моделі
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Навчання RandomForest...")
        rf.fit(X_train, y_train)
        
        # Оцінка якості
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        logger.info(f"R² train: {train_score:.4f}, R² test: {test_score:.4f}")
        
        # Feature importance
        importances = rf.feature_importances_
        
        # Створюємо DataFrame з результатами
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Топ-10 фічей за RandomForest:")
        for idx, row in rf_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.importance_results['random_forest'] = rf_importance
        
        return rf, rf_importance
    
    def calculate_permutation_importance(self, model, X_test, y_test, feature_names):
        """Розрахунок permutation importance"""
        logger.info("=== Розрахунок Permutation importance ===")
        
        # Розрахунок
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Створюємо DataFrame
        perm_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        logger.info(f"Топ-10 фічей за Permutation importance:")
        for idx, row in perm_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        self.importance_results['permutation'] = perm_df
        
        return perm_df
    
    def calculate_correlation_importance(self, X, y, feature_names):
        """Розрахунок важливості на основі кореляції"""
        logger.info("=== Розрахунок Correlation-based importance ===")
        
        correlations = []
        for col in X.columns:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations.append(abs(corr))
        
        corr_df = pd.DataFrame({
            'feature': feature_names,
            'correlation': correlations
        }).sort_values('correlation', ascending=False)
        
        logger.info(f"Топ-10 фічей за кореляцією:")
        for idx, row in corr_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['correlation']:.4f}")
        
        self.importance_results['correlation'] = corr_df
        
        return corr_df
    
    def aggregate_importance_scores(self):
        """Агрегація всіх методів оцінки важливості"""
        logger.info("=== Агрегація importance scores ===")
        
        # Нормалізація кожного методу
        all_features = set()
        normalized_scores = {}
        
        for method, df in self.importance_results.items():
            # Нормалізація від 0 до 1
            max_val = df.iloc[:, 1].max()
            min_val = df.iloc[:, 1].min()
            
            normalized = df.copy()
            normalized['normalized_score'] = (df.iloc[:, 1] - min_val) / (max_val - min_val)
            normalized_scores[method] = normalized
            all_features.update(df.iloc[:, 0].tolist())
        
        # Агрегація
        aggregate_scores = []
        
        for feature in all_features:
            scores = {}
            total_score = 0
            count = 0
            
            for method, df in normalized_scores.items():
                feature_row = df[df.iloc[:, 0] == feature]
                if not feature_row.empty:
                    score = feature_row['normalized_score'].values[0]
                    scores[f'{method}_score'] = score
                    total_score += score
                    count += 1
            
            if count > 0:
                scores['feature'] = feature
                scores['aggregate_score'] = total_score / count
                scores['n_methods'] = count
                aggregate_scores.append(scores)
        
        # Створюємо DataFrame
        aggregate_df = pd.DataFrame(aggregate_scores).sort_values('aggregate_score', ascending=False)
        
        logger.info(f"Топ-20 фічей за агрегованим score:")
        for idx, row in aggregate_df.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['aggregate_score']:.4f} ({row['n_methods']} методів)")
        
        self.top_features = aggregate_df.head(50)
        
        return aggregate_df
    
    def visualize_importance(self):
        """Візуалізація результатів"""
        logger.info("=== Візуалізація importance ===")
        
        # Створюємо фігуру з підграфіками
        fig = plt.figure(figsize=(20, 16))
        
        # 1. RandomForest importance (топ-20)
        ax1 = plt.subplot(2, 2, 1)
        rf_top20 = self.importance_results['random_forest'].head(20)
        ax1.barh(rf_top20['feature'], rf_top20['importance'])
        ax1.set_xlabel('Importance')
        ax1.set_title('Top-20 Features: RandomForest Importance', fontsize=14)
        ax1.invert_yaxis()
        
        # 2. Permutation importance (топ-20)
        ax2 = plt.subplot(2, 2, 2)
        perm_top20 = self.importance_results['permutation'].head(20)
        ax2.barh(perm_top20['feature'], perm_top20['importance_mean'])
        ax2.set_xlabel('Importance')
        ax2.set_title('Top-20 Features: Permutation Importance', fontsize=14)
        ax2.invert_yaxis()
        
        # 3. Correlation importance (топ-20)
        ax3 = plt.subplot(2, 2, 3)
        corr_top20 = self.importance_results['correlation'].head(20)
        ax3.barh(corr_top20['feature'], corr_top20['correlation'])
        ax3.set_xlabel('Correlation')
        ax3.set_title('Top-20 Features: Correlation with BTC Price', fontsize=14)
        ax3.invert_yaxis()
        
        # 4. Aggregate scores (топ-20)
        ax4 = plt.subplot(2, 2, 4)
        agg_top20 = self.top_features.head(20)
        colors = plt.cm.viridis(np.linspace(0, 1, 20))
        bars = ax4.barh(agg_top20['feature'], agg_top20['aggregate_score'], color=colors)
        ax4.set_xlabel('Aggregate Score')
        ax4.set_title('Top-20 Features: Aggregate Importance Score', fontsize=14)
        ax4.invert_yaxis()
        
        plt.tight_layout()
        output_path = 'feature_importance_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Візуалізація збережена: {output_path}")
        plt.close()
        
        # Створення матриці порівняння методів
        self.create_comparison_matrix()
        
        return output_path
    
    def create_comparison_matrix(self):
        """Створення матриці порівняння методів"""
        logger.info("Створення матриці порівняння методів")
        
        # Беремо топ-30 фічей
        top_features = self.top_features.head(30)['feature'].tolist()
        
        # Створюємо матрицю
        methods = ['random_forest', 'permutation', 'correlation']
        matrix_data = []
        
        for feature in top_features:
            row = [feature]
            for method in methods:
                df = self.importance_results[method]
                feature_row = df[df.iloc[:, 0] == feature]
                if not feature_row.empty:
                    # Знаходимо позицію в рейтингу
                    position = df.index[df.iloc[:, 0] == feature].tolist()[0] + 1
                    row.append(position)
                else:
                    row.append(np.nan)
            matrix_data.append(row)
        
        # Створюємо DataFrame
        comparison_df = pd.DataFrame(
            matrix_data,
            columns=['Feature', 'RF_Rank', 'Perm_Rank', 'Corr_Rank']
        )
        
        # Візуалізація
        plt.figure(figsize=(12, 10))
        
        # Підготовка даних для heatmap
        rank_matrix = comparison_df[['RF_Rank', 'Perm_Rank', 'Corr_Rank']].values
        
        # Створення heatmap
        sns.heatmap(
            rank_matrix.T,
            xticklabels=comparison_df['Feature'],
            yticklabels=['RandomForest', 'Permutation', 'Correlation'],
            cmap='YlOrRd_r',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Rank (lower is better)'}
        )
        
        plt.title('Feature Importance Rankings Comparison\n(Top-30 Features)', fontsize=14)
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        
        output_path = 'feature_importance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Матриця порівняння збережена: {output_path}")
        plt.close()
        
        return comparison_df
    
    def export_results(self):
        """Експорт результатів в CSV"""
        logger.info("=== Експорт результатів ===")
        
        # 1. Топ-50 агрегованих результатів
        top50_path = 'feature_importance_top50.csv'
        self.top_features.to_csv(top50_path, index=False)
        logger.info(f"Топ-50 фічей збережено: {top50_path}")
        
        # 2. Детальні результати кожного методу
        for method, df in self.importance_results.items():
            path = f'feature_importance_{method}.csv'
            df.to_csv(path, index=False)
            logger.info(f"{method} результати збережено: {path}")
        
        # 3. JSON з усіма результатами
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'total_features_analyzed': len(self.importance_results.get('random_forest', [])),
            'top_20_features': self.top_features.head(20).to_dict('records'),
            'summary': {
                'most_important_rf': self.importance_results['random_forest'].iloc[0]['feature'],
                'most_important_perm': self.importance_results['permutation'].iloc[0]['feature'],
                'most_correlated': self.importance_results['correlation'].iloc[0]['feature'],
                'most_important_overall': self.top_features.iloc[0]['feature']
            }
        }
        
        with open('feature_importance_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info("JSON результати збережено: feature_importance_results.json")
    
    def update_feature_metadata(self):
        """Оновлення таблиці feature_metadata"""
        logger.info("=== Оновлення feature_metadata ===")
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Оновлюємо importance_score для топ фічей
            for idx, row in self.top_features.iterrows():
                update_query = """
                UPDATE feature_metadata 
                SET importance_score = %s,
                    last_updated = NOW()
                WHERE feature_name = %s
                """
                
                cursor.execute(update_query, (
                    float(row['aggregate_score']),
                    row['feature']
                ))
            
            conn.commit()
            logger.info(f"Оновлено {len(self.top_features)} записів в feature_metadata")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Помилка оновлення metadata: {str(e)}")
    
    def generate_report(self):
        """Генерація звіту"""
        logger.info("=== Генерація звіту ===")
        
        report = "# Feature Importance Analysis Report\n\n"
        report += f"**Дата аналізу:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Загальна інформація
        report += "## Загальна інформація\n"
        report += f"- Проаналізовано фічей: {len(self.importance_results.get('random_forest', []))}\n"
        report += f"- Методів оцінки: 3 (RandomForest, Permutation, Correlation)\n"
        report += f"- Агреговано топ-фічей: {len(self.top_features)}\n\n"
        
        # Топ-20 найважливіших фічей
        report += "## Топ-20 найважливіших фічей (агрегований score)\n\n"
        report += "| Rank | Feature | Aggregate Score | RF Importance | Permutation | Correlation |\n"
        report += "|------|---------|----------------|---------------|-------------|-------------|\n"
        
        for idx, row in self.top_features.head(20).iterrows():
            rf_score = perm_score = corr_score = "N/A"
            
            # Знаходимо scores з різних методів
            if 'random_forest_score' in row:
                rf_score = f"{row['random_forest_score']:.4f}"
            if 'permutation_score' in row:
                perm_score = f"{row['permutation_score']:.4f}"
            if 'correlation_score' in row:
                corr_score = f"{row['correlation_score']:.4f}"
            
            report += f"| {idx+1} | {row['feature']} | {row['aggregate_score']:.4f} | "
            report += f"{rf_score} | {perm_score} | {corr_score} |\n"
        
        # Інсайти
        report += "\n## Ключові інсайти\n\n"
        
        # Найважливіші фічі по кожному методу
        report += "### Топ-5 по кожному методу:\n\n"
        
        report += "**RandomForest:**\n"
        for idx, row in self.importance_results['random_forest'].head(5).iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
        
        report += "\n**Permutation:**\n"
        for idx, row in self.importance_results['permutation'].head(5).iterrows():
            report += f"- {row['feature']}: {row['importance_mean']:.4f}\n"
        
        report += "\n**Correlation:**\n"
        for idx, row in self.importance_results['correlation'].head(5).iterrows():
            report += f"- {row['feature']}: {row['correlation']:.4f}\n"
        
        # Рекомендації
        report += "\n## Рекомендації\n\n"
        report += "1. Використовувати топ-20 фічей для побудови моделей\n"
        report += "2. Особливу увагу приділити фічам з високим aggregate score (>0.7)\n"
        report += "3. Розглянути можливість видалення фічей з низькою важливістю\n"
        report += "4. Протестувати моделі з різною кількістю фічей (10, 20, 30, 50)\n"
        
        # Збереження
        with open('feature_importance_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Звіт збережено: feature_importance_report.md")
        
        return report
    
    def run_full_analysis(self):
        """Запуск повного аналізу"""
        logger.info("=== FEATURE IMPORTANCE ANALYSIS START ===")
        
        try:
            # 1. Завантаження даних
            df, metadata = self.load_data_with_features()
            
            # 2. Підготовка даних
            target_col = 'btc_price'
            feature_cols = [col for col in df.columns if col != target_col]
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Масштабування
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 3. RandomForest importance
            rf_model, rf_importance = self.calculate_rf_importance(X_scaled, y, feature_cols)
            
            # 4. Permutation importance
            perm_importance = self.calculate_permutation_importance(
                rf_model, X_test, y_test, feature_cols
            )
            
            # 5. Correlation importance
            corr_importance = self.calculate_correlation_importance(X_scaled, y, feature_cols)
            
            # 6. Агрегація результатів
            aggregate_df = self.aggregate_importance_scores()
            
            # 7. Візуалізація
            viz_path = self.visualize_importance()
            
            # 8. Експорт результатів
            self.export_results()
            
            # 9. Оновлення metadata
            self.update_feature_metadata()
            
            # 10. Генерація звіту
            report = self.generate_report()
            
            logger.info("=== FEATURE IMPORTANCE ANALYSIS COMPLETED ===")
            
            return {
                'success': True,
                'top_features': self.top_features,
                'visualization': viz_path,
                'report': 'feature_importance_report.md'
            }
            
        except Exception as e:
            logger.error(f"Критична помилка аналізу: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Головна функція"""
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.run_full_analysis()
    
    if results['success']:
        print("\n✅ Feature Importance аналіз успішно завершено!")
        print(f"Візуалізація: {results['visualization']}")
        print(f"Звіт: {results['report']}")
        print(f"\nТоп-5 найважливіших фічей:")
        for idx, row in results['top_features'].head(5).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['aggregate_score']:.4f}")
    else:
        print(f"\n❌ Помилка аналізу: {results['error']}")


if __name__ == "__main__":
    main()