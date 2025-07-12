#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Granger Causality Analysis для GT14 v14.2
Покращений аналіз з F-статистикою, візуалізацією та різними лагами
"""

import pandas as pd
import numpy as np
import mysql.connector
from statsmodels.tsa.stattools import grangercausalitytests
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
        logging.FileHandler('granger_causality_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GrangerCausalityEnhanced:
    """Покращений аналіз Granger causality з повним логуванням"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        self.results = {}
        self.f_statistics = {}
        self.p_values = {}
        logger.info("Ініціалізація GrangerCausalityEnhanced")
        
    def load_data(self):
        """Завантаження даних для аналізу"""
        logger.info("=== Завантаження даних для Granger causality ===")
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Основні змінні для аналізу
            query = """
            SELECT timestamp, btc_price, whale_volume_usd, net_flow, 
                   whale_activity, exchange_inflow, exchange_outflow,
                   fear_greed_index
            FROM whale_hourly_complete
            WHERE btc_price > 0 AND whale_volume_usd > 0
            ORDER BY timestamp
            """
            
            df = pd.read_sql(query, conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Завантажено {len(df)} записів")
            logger.info(f"Змінні: {list(df.columns)}")
            
            # Перевірка стаціонарності та диференціювання
            df_diff = df.diff().dropna()
            
            conn.close()
            
            return df, df_diff
            
        except Exception as e:
            logger.error(f"Помилка завантаження даних: {str(e)}")
            raise
    
    def test_granger_causality(self, data, x_var, y_var, max_lag=5):
        """Тест Granger causality для пари змінних"""
        logger.info(f"Тестування: {x_var} → {y_var}")
        
        try:
            # Підготовка даних
            test_data = data[[y_var, x_var]].dropna()
            
            if len(test_data) < 50:
                logger.warning(f"Недостатньо даних для тесту ({len(test_data)} записів)")
                return None
            
            # Запуск тесту для різних лагів
            results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            
            # Збір результатів
            test_results = {
                'x_var': x_var,
                'y_var': y_var,
                'lags': {},
                'best_lag': None,
                'best_f_stat': 0,
                'best_p_value': 1.0,
                'significant': False
            }
            
            for lag in range(1, max_lag + 1):
                # F-тест
                f_test = results[lag][0]['ssr_ftest']
                f_stat = f_test[0]
                p_value = f_test[1]
                
                # Chi2-тест
                chi2_test = results[lag][0]['ssr_chi2test']
                chi2_stat = chi2_test[0]
                chi2_p = chi2_test[1]
                
                test_results['lags'][lag] = {
                    'f_statistic': float(f_stat),
                    'f_p_value': float(p_value),
                    'chi2_statistic': float(chi2_stat),
                    'chi2_p_value': float(chi2_p),
                    'significant': p_value < 0.05
                }
                
                # Визначення найкращого лагу
                if p_value < test_results['best_p_value']:
                    test_results['best_lag'] = lag
                    test_results['best_f_stat'] = float(f_stat)
                    test_results['best_p_value'] = float(p_value)
                    test_results['significant'] = p_value < 0.05
            
            logger.info(f"  Найкращий лаг: {test_results['best_lag']}, "
                       f"F-stat: {test_results['best_f_stat']:.4f}, "
                       f"p-value: {test_results['best_p_value']:.4f}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Помилка тесту {x_var} → {y_var}: {str(e)}")
            return None
    
    def run_pairwise_analysis(self, data, variables, max_lag=5):
        """Запуск попарного аналізу для всіх змінних"""
        logger.info(f"=== Попарний аналіз Granger causality (max_lag={max_lag}) ===")
        
        results = {}
        
        # Тестуємо всі пари змінних
        for i, x_var in enumerate(variables):
            for j, y_var in enumerate(variables):
                if i != j:  # Не тестуємо змінну саму на себе
                    pair_key = f"{x_var} → {y_var}"
                    logger.info(f"\nТестування пари: {pair_key}")
                    
                    result = self.test_granger_causality(data, x_var, y_var, max_lag)
                    
                    if result:
                        results[pair_key] = result
                        
                        # Зберігаємо F-статистику та p-value
                        if x_var not in self.f_statistics:
                            self.f_statistics[x_var] = {}
                            self.p_values[x_var] = {}
                        
                        self.f_statistics[x_var][y_var] = result['best_f_stat']
                        self.p_values[x_var][y_var] = result['best_p_value']
        
        return results
    
    def create_causality_matrix(self, variables):
        """Створення матриці причинності"""
        logger.info("=== Створення матриці причинності ===")
        
        n = len(variables)
        f_matrix = np.zeros((n, n))
        p_matrix = np.ones((n, n))
        
        for i, x_var in enumerate(variables):
            for j, y_var in enumerate(variables):
                if x_var in self.f_statistics and y_var in self.f_statistics[x_var]:
                    f_matrix[i, j] = self.f_statistics[x_var][y_var]
                    p_matrix[i, j] = self.p_values[x_var][y_var]
        
        return f_matrix, p_matrix
    
    def visualize_results(self, variables):
        """Візуалізація результатів Granger causality"""
        logger.info("=== Візуалізація результатів ===")
        
        try:
            # Створюємо матриці
            f_matrix, p_matrix = self.create_causality_matrix(variables)
            
            # Створюємо фігуру з двома підграфіками
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 1. Матриця F-статистик
            mask = np.eye(len(variables), dtype=bool)  # Маска для діагоналі
            
            sns.heatmap(f_matrix, 
                       mask=mask,
                       annot=True, 
                       fmt='.2f',
                       xticklabels=variables,
                       yticklabels=variables,
                       cmap='YlOrRd',
                       center=0,
                       ax=ax1,
                       cbar_kws={'label': 'F-statistic'})
            
            ax1.set_title('Granger Causality F-Statistics\n(рядок → колонка)', fontsize=14)
            ax1.set_xlabel('Залежна змінна (Effect)', fontsize=12)
            ax1.set_ylabel('Незалежна змінна (Cause)', fontsize=12)
            
            # 2. Матриця p-values з позначенням значущих
            significance_matrix = p_matrix.copy()
            significance_matrix[p_matrix >= 0.05] = np.nan
            
            sns.heatmap(p_matrix,
                       mask=mask,
                       annot=True,
                       fmt='.3f',
                       xticklabels=variables,
                       yticklabels=variables,
                       cmap='RdYlGn_r',
                       vmin=0,
                       vmax=0.1,
                       ax=ax2,
                       cbar_kws={'label': 'p-value'})
            
            # Додаємо зірочки для значущих результатів
            for i in range(len(variables)):
                for j in range(len(variables)):
                    if i != j and p_matrix[i, j] < 0.05:
                        ax2.text(j + 0.5, i + 0.5, '*', 
                                ha='center', va='center',
                                fontsize=16, fontweight='bold')
                        if p_matrix[i, j] < 0.01:
                            ax2.text(j + 0.5, i + 0.7, '*', 
                                    ha='center', va='center',
                                    fontsize=16, fontweight='bold')
            
            ax2.set_title('Granger Causality p-values\n(* p<0.05, ** p<0.01)', fontsize=14)
            ax2.set_xlabel('Залежна змінна (Effect)', fontsize=12)
            ax2.set_ylabel('Незалежна змінна (Cause)', fontsize=12)
            
            plt.tight_layout()
            output_path = 'granger_causality_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Матриця збережена: {output_path}")
            
            plt.close()
            
            # Створення графіку сили зв'язків
            self.create_strength_plot()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Помилка візуалізації: {str(e)}")
            raise
    
    def create_strength_plot(self):
        """Створення графіку сили причинних зв'язків"""
        logger.info("Створення графіку сили зв'язків")
        
        # Збираємо всі значущі зв'язки
        significant_links = []
        
        for pair, result in self.results.items():
            if result['significant']:
                significant_links.append({
                    'pair': pair,
                    'f_stat': result['best_f_stat'],
                    'p_value': result['best_p_value'],
                    'lag': result['best_lag']
                })
        
        if not significant_links:
            logger.warning("Немає значущих причинних зв'язків")
            return
        
        # Сортуємо за F-статистикою
        significant_links.sort(key=lambda x: x['f_stat'], reverse=True)
        
        # Візуалізація топ-10
        top_links = significant_links[:10]
        
        plt.figure(figsize=(12, 8))
        
        pairs = [link['pair'] for link in top_links]
        f_stats = [link['f_stat'] for link in top_links]
        lags = [link['lag'] for link in top_links]
        
        bars = plt.barh(pairs, f_stats)
        
        # Колір за лагом
        colors = plt.cm.viridis(np.linspace(0, 1, 5))
        for bar, lag in zip(bars, lags):
            bar.set_color(colors[lag-1])
        
        plt.xlabel('F-statistic', fontsize=12)
        plt.title('Топ-10 найсильніших причинних зв\'язків\n(Granger Causality)', fontsize=14)
        
        # Легенда для лагів
        from matplotlib.patches import Rectangle
        legend_elements = [Rectangle((0,0),1,1, fc=colors[i], label=f'Lag {i+1}') 
                          for i in range(5)]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        output_path = 'granger_causality_strength.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Графік сили збережено: {output_path}")
        
        plt.close()
    
    def export_to_csv(self):
        """Експорт результатів в CSV"""
        logger.info("=== Експорт результатів в CSV ===")
        
        # Підготовка даних для експорту
        export_data = []
        
        for pair, result in self.results.items():
            row = {
                'causality_pair': pair,
                'x_variable': result['x_var'],
                'y_variable': result['y_var'],
                'best_lag': result['best_lag'],
                'f_statistic': result['best_f_stat'],
                'p_value': result['best_p_value'],
                'significant': result['significant']
            }
            
            # Додаємо результати для кожного лагу
            for lag in range(1, 6):
                if lag in result['lags']:
                    row[f'lag_{lag}_f_stat'] = result['lags'][lag]['f_statistic']
                    row[f'lag_{lag}_p_value'] = result['lags'][lag]['f_p_value']
            
            export_data.append(row)
        
        # Створюємо DataFrame та експортуємо
        df_export = pd.DataFrame(export_data)
        
        # Сортуємо за значущістю та F-статистикою
        df_export = df_export.sort_values(['significant', 'f_statistic'], 
                                        ascending=[False, False])
        
        output_path = 'granger_causality_results.csv'
        df_export.to_csv(output_path, index=False)
        logger.info(f"Результати експортовано: {output_path}")
        
        # Створюємо окремий файл для значущих зв'язків
        significant_df = df_export[df_export['significant'] == True]
        if len(significant_df) > 0:
            sig_output = 'granger_causality_significant.csv'
            significant_df.to_csv(sig_output, index=False)
            logger.info(f"Значущі зв'язки експортовано: {sig_output}")
        
        return df_export
    
    def generate_report(self):
        """Генерація текстового звіту"""
        logger.info("=== Генерація звіту ===")
        
        report = "# Granger Causality Analysis Report\n\n"
        report += f"**Дата аналізу:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Загальна статистика
        total_tests = len(self.results)
        significant_tests = sum(1 for r in self.results.values() if r['significant'])
        
        report += "## Загальна статистика\n"
        report += f"- Всього тестів: {total_tests}\n"
        report += f"- Значущих зв'язків (p<0.05): {significant_tests}\n"
        report += f"- Відсоток значущих: {(significant_tests/total_tests*100):.1f}%\n\n"
        
        # Найсильніші зв'язки
        report += "## Найсильніші причинні зв'язки\n\n"
        
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['best_f_stat'], 
                              reverse=True)
        
        report += "| Зв'язок | F-stat | p-value | Lag | Значущий |\n"
        report += "|---------|--------|---------|-----|----------|\n"
        
        for pair, result in sorted_results[:15]:
            if result['significant']:
                report += f"| {pair} | {result['best_f_stat']:.4f} | "
                report += f"{result['best_p_value']:.4f} | {result['best_lag']} | ✓ |\n"
        
        # Аналіз по змінних
        report += "\n## Аналіз по змінних\n\n"
        
        # Які змінні найбільше впливають
        cause_counts = {}
        for pair, result in self.results.items():
            if result['significant']:
                cause = result['x_var']
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
        
        if cause_counts:
            report += "### Змінні з найбільшим впливом (causes):\n"
            for var, count in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True):
                report += f"- **{var}**: впливає на {count} змінних\n"
        
        # Які змінні найбільше залежать
        effect_counts = {}
        for pair, result in self.results.items():
            if result['significant']:
                effect = result['y_var']
                effect_counts[effect] = effect_counts.get(effect, 0) + 1
        
        if effect_counts:
            report += "\n### Змінні з найбільшою залежністю (effects):\n"
            for var, count in sorted(effect_counts.items(), key=lambda x: x[1], reverse=True):
                report += f"- **{var}**: залежить від {count} змінних\n"
        
        report += "\n## Рекомендації\n"
        report += "1. Використовувати змінні з сильними причинними зв'язками для прогнозування\n"
        report += "2. Враховувати оптимальні лаги при побудові моделей\n"
        report += "3. Звернути увагу на двонаправлені зв'язки\n"
        
        # Збереження звіту
        with open('granger_causality_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Звіт збережено: granger_causality_report.md")
        
        return report
    
    def analyze_granger_causality(self, df_input, max_lag=5):
        """Аналіз Granger causality з переданим DataFrame"""
        logger.info("=== GRANGER CAUSALITY ENHANCED ANALYSIS (з переданим DataFrame) ===")
        
        try:
            # Використовуємо переданий DataFrame
            logger.info(f"Отримано DataFrame: {df_input.shape[0]} рядків, {df_input.shape[1]} колонок")
            logger.info(f"Доступні колонки: {list(df_input.columns)[:20]}...")  # Показуємо перші 20
            
            # Базові змінні які завжди аналізуємо
            base_variables = ['btc_price', 'whale_volume_usd', 'net_flow', 
                            'whale_activity', 'exchange_inflow', 'exchange_outflow']
            
            # Додаємо оптимальні фічі якщо вони є
            additional_features = []
            for col in df_input.columns:
                if col not in base_variables and col not in ['timestamp', 'fear_greed_classification', 
                                                            'market_sentiment', 'SP500', 'VIX', 
                                                            'GOLD', 'NASDAQ', 'OIL_WTI']:
                    # Додаємо числові колонки
                    if df_input[col].dtype in ['float64', 'int64']:
                        additional_features.append(col)
            
            # Обмежуємо кількість додаткових фічей для аналізу (топ-10)
            if len(additional_features) > 10:
                logger.info(f"Знайдено {len(additional_features)} додаткових фічей, використовуємо топ-10")
                # Вибираємо топ-10 за варіацією
                feature_vars = {col: df_input[col].var() for col in additional_features 
                               if df_input[col].notna().sum() > 100}
                top_features = sorted(feature_vars.items(), key=lambda x: x[1], reverse=True)[:10]
                additional_features = [f[0] for f in top_features]
            
            # Всі змінні для аналізу
            variables = base_variables + additional_features
            logger.info(f"Змінні для аналізу ({len(variables)}): {variables}")
            
            # Підготовка даних - диференціювання
            df_for_analysis = df_input[variables].copy()
            df_diff = df_for_analysis.diff().dropna()
            
            # Попарний аналіз
            self.results = self.run_pairwise_analysis(df_diff, variables, max_lag)
            
            # Візуалізація
            matrix_path = self.visualize_results(variables)
            
            # Експорт в CSV
            export_df = self.export_to_csv()
            
            # Генерація звіту
            report = self.generate_report()
            
            # Збереження результатів в JSON
            json_results = {
                'timestamp': datetime.now().isoformat(),
                'total_pairs': len(self.results),
                'significant_pairs': sum(1 for r in self.results.values() if r['significant']),
                'max_lag': max_lag,
                'variables_analyzed': len(variables),
                'additional_features': len(additional_features)
            }
            
            logger.info(f"✅ Аналіз завершено: {json_results['significant_pairs']}/{json_results['total_pairs']} значущих зв'язків")
            
            return json_results
            
        except Exception as e:
            logger.error(f"Помилка аналізу: {str(e)}")
            return None
    
    def run_full_analysis(self, max_lag=5):
        """Запуск повного аналізу"""
        logger.info("=== GRANGER CAUSALITY ENHANCED ANALYSIS ===")
        
        try:
            # 1. Завантаження даних
            df, df_diff = self.load_data()
            
            # 2. Визначення змінних для аналізу
            variables = ['btc_price', 'whale_volume_usd', 'net_flow', 
                        'whale_activity', 'exchange_inflow', 'exchange_outflow']
            
            # 3. Попарний аналіз
            self.results = self.run_pairwise_analysis(df_diff, variables, max_lag)
            
            # 4. Візуалізація
            matrix_path = self.visualize_results(variables)
            
            # 5. Експорт в CSV
            export_df = self.export_to_csv()
            
            # 6. Генерація звіту
            report = self.generate_report()
            
            # 7. Збереження результатів в JSON
            # Конвертуємо bool значення в результатах для JSON
            json_results = {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'significant_tests': sum(1 for r in self.results.values() if r['significant']),
                'max_lag': max_lag,
                'results': {}
            }
            
            # Конвертуємо результати для JSON
            for pair, result in self.results.items():
                json_result = {
                    'x_var': result['x_var'],
                    'y_var': result['y_var'],
                    'best_lag': result['best_lag'],
                    'best_f_stat': result['best_f_stat'],
                    'best_p_value': result['best_p_value'],
                    'significant': bool(result['significant']),  # Конвертуємо в bool
                    'lags': {}
                }
                
                # Конвертуємо лаги
                for lag, lag_data in result['lags'].items():
                    json_result['lags'][str(lag)] = {
                        'f_statistic': lag_data['f_statistic'],
                        'f_p_value': lag_data['f_p_value'],
                        'chi2_statistic': lag_data['chi2_statistic'],
                        'chi2_p_value': lag_data['chi2_p_value'],
                        'significant': bool(lag_data['significant'])
                    }
                
                json_results['results'][pair] = json_result
            
            with open('granger_causality_results.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info("=== АНАЛІЗ ЗАВЕРШЕНО УСПІШНО ===")
            
            return {
                'success': True,
                'results': self.results,
                'matrix_path': matrix_path,
                'export_path': 'granger_causality_results.csv',
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Критична помилка аналізу: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Головна функція"""
    analyzer = GrangerCausalityEnhanced()
    results = analyzer.run_full_analysis(max_lag=5)
    
    if results['success']:
        print("\n✅ Granger Causality аналіз успішно завершено!")
        print(f"Матриця: {results['matrix_path']}")
        print(f"CSV: {results['export_path']}")
        print("Звіт: granger_causality_report.md")
    else:
        print(f"\n❌ Помилка аналізу: {results['error']}")


if __name__ == "__main__":
    main()