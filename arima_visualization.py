#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA Visualization Module для GT14 v14.2
Візуалізація прогнозів ARIMA з повним покриттям тестами та логами
"""

import pandas as pd
import numpy as np
import mysql.connector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arima_visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ARIMAVisualization:
    """Клас для візуалізації ARIMA прогнозів з повним логуванням"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        logger.info("Ініціалізація ARIMAVisualization")
        
    def load_data(self):
        """Завантаження історичних даних та прогнозів"""
        logger.info("=== Завантаження даних для візуалізації ===")
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Історичні дані (останні 14 днів для кращої варіативності)
            historical_query = """
            SELECT timestamp, btc_price
            FROM whale_hourly_complete
            WHERE btc_price > 0
            ORDER BY timestamp DESC
            LIMIT 336
            """
            
            historical_data = pd.read_sql(historical_query, conn)
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
            historical_data = historical_data.sort_values('timestamp')
            
            logger.info(f"Завантажено {len(historical_data)} історичних записів")
            
            # Прогнози ARIMA
            forecast_query = """
            SELECT forecast_timestamp, forecast_price, lower_ci, upper_ci
            FROM arima_forecasts
            ORDER BY forecast_timestamp
            """
            
            forecast_data = pd.read_sql(forecast_query, conn)
            forecast_data['forecast_timestamp'] = pd.to_datetime(forecast_data['forecast_timestamp'])
            
            logger.info(f"Завантажено {len(forecast_data)} прогнозних записів")
            
            # Модель інформація
            model_query = """
            SELECT model_name, mape, mae, created_at
            FROM arima_models
            ORDER BY created_at DESC
            LIMIT 1
            """
            
            model_info = pd.read_sql(model_query, conn)
            
            conn.close()
            
            if len(model_info) > 0:
                logger.info(f"Модель: {model_info['model_name'].iloc[0]}, MAPE: {model_info['mape'].iloc[0]}%")
            
            return historical_data, forecast_data, model_info
            
        except Exception as e:
            logger.error(f"Помилка завантаження даних: {str(e)}")
            raise
    
    def create_static_plot(self, historical_data, forecast_data, model_info):
        """Створення статичного PNG графіку"""
        logger.info("=== Створення статичного графіку ===")
        
        try:
            # Налаштування стилю
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Історичні дані
            ax.plot(historical_data['timestamp'], 
                   historical_data['btc_price'], 
                   label='Історичні дані', 
                   color='blue', 
                   linewidth=2)
            
            # Прогноз
            if len(forecast_data) > 0:
                ax.plot(forecast_data['forecast_timestamp'], 
                       forecast_data['forecast_price'], 
                       label='ARIMA прогноз', 
                       color='red', 
                       linewidth=2,
                       linestyle='--')
                
                # Довірчі інтервали
                ax.fill_between(forecast_data['forecast_timestamp'],
                               forecast_data['lower_ci'],
                               forecast_data['upper_ci'],
                               color='red', 
                               alpha=0.2,
                               label='95% довірчий інтервал')
            
            # З'єднання історичних даних з прогнозом
            if len(historical_data) > 0 and len(forecast_data) > 0:
                last_historical = historical_data.iloc[-1]
                first_forecast = forecast_data.iloc[0]
                
                ax.plot([last_historical['timestamp'], first_forecast['forecast_timestamp']],
                       [last_historical['btc_price'], first_forecast['forecast_price']],
                       color='gray', 
                       linewidth=1, 
                       linestyle=':')
            
            # Форматування
            ax.set_title('ARIMA Прогноз Bitcoin - GT14 v14.2', fontsize=16, fontweight='bold')
            ax.set_xlabel('Час', fontsize=12)
            ax.set_ylabel('Ціна BTC ($)', fontsize=12)
            
            # Форматування дат на осі X
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            plt.xticks(rotation=45)
            
            # Легенда
            ax.legend(loc='upper left')
            
            # Текстова інформація про модель
            if len(model_info) > 0:
                model_text = f"Модель: {model_info['model_name'].iloc[0]}\n"
                model_text += f"MAPE: {model_info['mape'].iloc[0]:.2f}%\n"
                model_text += f"MAE: ${model_info['mae'].iloc[0]:.2f}"
                
                ax.text(0.02, 0.98, model_text,
                       transform=ax.transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Сітка
            ax.grid(True, alpha=0.3)
            
            # Збереження
            plt.tight_layout()
            output_path = 'arima_forecast_visualization.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Статичний графік збережено: {output_path}")
            
            plt.close()
            
            # Статистика візуалізації
            stats = {
                'historical_points': len(historical_data),
                'forecast_points': len(forecast_data),
                'min_price': float(historical_data['btc_price'].min()),
                'max_price': float(historical_data['btc_price'].max()),
                'last_price': float(historical_data['btc_price'].iloc[-1]) if len(historical_data) > 0 else 0
            }
            
            if len(forecast_data) > 0:
                stats['forecast_min'] = float(forecast_data['forecast_price'].min())
                stats['forecast_max'] = float(forecast_data['forecast_price'].max())
                stats['forecast_24h'] = float(forecast_data['forecast_price'].iloc[-1])
                stats['change_24h'] = ((stats['forecast_24h'] / stats['last_price'] - 1) * 100) if stats['last_price'] > 0 else 0
            
            logger.info(f"Статистика: {json.dumps(stats, indent=2)}")
            
            return output_path, stats
            
        except Exception as e:
            logger.error(f"Помилка створення статичного графіку: {str(e)}")
            raise
    
    def create_interactive_plot(self, historical_data, forecast_data, model_info):
        """Створення інтерактивного Plotly графіку"""
        logger.info("=== Створення інтерактивного графіку ===")
        
        try:
            fig = go.Figure()
            
            # Історичні дані
            fig.add_trace(go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['btc_price'],
                mode='lines',
                name='Історичні дані',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Час:</b> %{x}<br><b>Ціна:</b> $%{y:,.2f}<extra></extra>'
            ))
            
            # Прогноз
            if len(forecast_data) > 0:
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast_timestamp'],
                    y=forecast_data['forecast_price'],
                    mode='lines',
                    name='ARIMA прогноз',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='<b>Час:</b> %{x}<br><b>Прогноз:</b> $%{y:,.2f}<extra></extra>'
                ))
                
                # Верхня межа довірчого інтервалу
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast_timestamp'],
                    y=forecast_data['upper_ci'],
                    mode='lines',
                    name='Верхня межа (95%)',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Нижня межа довірчого інтервалу
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast_timestamp'],
                    y=forecast_data['lower_ci'],
                    mode='lines',
                    name='Нижня межа (95%)',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Додаємо в легенду довірчий інтервал
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    name='95% довірчий інтервал',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
            
            # Налаштування макету
            fig.update_layout(
                title={
                    'text': 'ARIMA Прогноз Bitcoin - GT14 v14.2 (Інтерактивний)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Час',
                yaxis_title='Ціна BTC ($)',
                hovermode='x unified',
                template='plotly_white',
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Додаємо анотацію з інформацією про модель
            if len(model_info) > 0:
                annotation_text = f"Модель: {model_info['model_name'].iloc[0]}<br>"
                annotation_text += f"MAPE: {model_info['mape'].iloc[0]:.2f}%<br>"
                annotation_text += f"MAE: ${model_info['mae'].iloc[0]:.2f}"
                
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    text=annotation_text,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    font=dict(size=12),
                    align="left"
                )
            
            # Кнопки для зміни діапазону
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1д", step="day", stepmode="backward"),
                        dict(count=3, label="3д", step="day", stepmode="backward"),
                        dict(count=7, label="7д", step="day", stepmode="backward"),
                        dict(step="all", label="Все")
                    ])
                )
            )
            
            # Збереження
            output_path = 'arima_forecast_interactive.html'
            fig.write_html(output_path)
            logger.info(f"Інтерактивний графік збережено: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Помилка створення інтерактивного графіку: {str(e)}")
            raise
    
    def create_comparison_plot(self, historical_data, forecast_data):
        """Створення графіку порівняння факт vs прогноз (якщо є перетин)"""
        logger.info("=== Створення графіку порівняння ===")
        
        try:
            # Перевіряємо чи є дані для порівняння
            if len(forecast_data) == 0:
                logger.warning("Немає прогнозних даних для порівняння")
                return None
            
            # Знаходимо перетин між історичними даними та прогнозом
            min_forecast_time = forecast_data['forecast_timestamp'].min()
            overlap_data = historical_data[historical_data['timestamp'] >= min_forecast_time]
            
            if len(overlap_data) == 0:
                logger.info("Немає перетину між історичними даними та прогнозом")
                return None
            
            # Створюємо графік порівняння
            plt.figure(figsize=(12, 6))
            
            # Фактичні дані в зоні прогнозу
            plt.plot(overlap_data['timestamp'], 
                    overlap_data['btc_price'], 
                    label='Фактична ціна', 
                    color='blue', 
                    linewidth=2,
                    marker='o')
            
            # Прогнозні дані
            overlap_forecast = forecast_data[
                forecast_data['forecast_timestamp'].isin(overlap_data['timestamp'])
            ]
            
            if len(overlap_forecast) > 0:
                plt.plot(overlap_forecast['forecast_timestamp'], 
                        overlap_forecast['forecast_price'], 
                        label='Прогноз', 
                        color='red', 
                        linewidth=2,
                        marker='x')
                
                # Розрахунок помилок
                errors = []
                for _, row in overlap_forecast.iterrows():
                    actual = overlap_data[
                        overlap_data['timestamp'] == row['forecast_timestamp']
                    ]['btc_price'].values
                    
                    if len(actual) > 0:
                        error = abs(actual[0] - row['forecast_price'])
                        percentage_error = (error / actual[0]) * 100
                        errors.append({
                            'time': row['forecast_timestamp'],
                            'actual': actual[0],
                            'forecast': row['forecast_price'],
                            'error': error,
                            'percentage_error': percentage_error
                        })
                
                if errors:
                    avg_error = np.mean([e['percentage_error'] for e in errors])
                    logger.info(f"Середня помилка прогнозу: {avg_error:.2f}%")
                    
                    # Додаємо текст з помилкою
                    plt.text(0.02, 0.02, f'Середня помилка: {avg_error:.2f}%',
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            plt.title('Порівняння: Факт vs Прогноз', fontsize=14)
            plt.xlabel('Час')
            plt.ylabel('Ціна BTC ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            output_path = 'arima_comparison_fact_vs_forecast.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Графік порівняння збережено: {output_path}")
            
            plt.close()
            
            return output_path, errors if 'errors' in locals() else None
            
        except Exception as e:
            logger.error(f"Помилка створення графіку порівняння: {str(e)}")
            return None
    
    def generate_report(self, stats, errors=None):
        """Генерація текстового звіту"""
        logger.info("=== Генерація звіту ===")
        
        try:
            report = "# ARIMA Візуалізація - Звіт\n\n"
            report += f"**Дата генерації:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report += "## Статистика даних\n"
            report += f"- Історичних точок: {stats.get('historical_points', 0)}\n"
            report += f"- Прогнозних точок: {stats.get('forecast_points', 0)}\n"
            report += f"- Діапазон історичних цін: ${stats.get('min_price', 0):,.2f} - ${stats.get('max_price', 0):,.2f}\n"
            report += f"- Остання ціна: ${stats.get('last_price', 0):,.2f}\n"
            
            if 'forecast_24h' in stats:
                report += f"\n## Прогноз\n"
                report += f"- Прогноз на 24 години: ${stats.get('forecast_24h', 0):,.2f}\n"
                report += f"- Зміна за 24 години: {stats.get('change_24h', 0):+.2f}%\n"
                report += f"- Діапазон прогнозу: ${stats.get('forecast_min', 0):,.2f} - ${stats.get('forecast_max', 0):,.2f}\n"
            
            if errors:
                report += f"\n## Точність прогнозу (на історичних даних)\n"
                report += f"- Кількість порівнянь: {len(errors)}\n"
                report += f"- Середня помилка: {np.mean([e['percentage_error'] for e in errors]):.2f}%\n"
                report += f"- Мінімальна помилка: {min(e['percentage_error'] for e in errors):.2f}%\n"
                report += f"- Максимальна помилка: {max(e['percentage_error'] for e in errors):.2f}%\n"
            
            report += "\n## Створені файли\n"
            report += "- arima_forecast_visualization.png\n"
            report += "- arima_forecast_interactive.html\n"
            if errors:
                report += "- arima_comparison_fact_vs_forecast.png\n"
            
            # Збереження звіту
            with open('arima_visualization_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info("Звіт збережено: arima_visualization_report.md")
            
            return report
            
        except Exception as e:
            logger.error(f"Помилка генерації звіту: {str(e)}")
            raise
    
    def run_visualization(self):
        """Запуск повної візуалізації"""
        logger.info("=== ARIMA VISUALIZATION START ===")
        
        try:
            # 1. Завантаження даних
            historical_data, forecast_data, model_info = self.load_data()
            
            # 2. Створення статичного графіку
            png_path, stats = self.create_static_plot(historical_data, forecast_data, model_info)
            
            # 3. Створення інтерактивного графіку
            html_path = self.create_interactive_plot(historical_data, forecast_data, model_info)
            
            # 4. Створення графіку порівняння
            comparison_result = self.create_comparison_plot(historical_data, forecast_data)
            errors = comparison_result[1] if comparison_result else None
            
            # 5. Генерація звіту
            report = self.generate_report(stats, errors)
            
            logger.info("=== ARIMA VISUALIZATION COMPLETED ===")
            
            return {
                'success': True,
                'png_path': png_path,
                'html_path': html_path,
                'comparison_path': comparison_result[0] if comparison_result else None,
                'stats': stats,
                'errors': errors,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Критична помилка візуалізації: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Головна функція"""
    visualizer = ARIMAVisualization()
    results = visualizer.run_visualization()
    
    if results['success']:
        print("\n✅ Візуалізація успішно завершена!")
        print(f"PNG: {results['png_path']}")
        print(f"HTML: {results['html_path']}")
        if results['comparison_path']:
            print(f"Comparison: {results['comparison_path']}")
    else:
        print(f"\n❌ Помилка візуалізації: {results['error']}")


if __name__ == "__main__":
    main()