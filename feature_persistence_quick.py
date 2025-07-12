#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Feature Persistence для швидкого збереження фічей
Оптимізована версія
"""

import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturePersistenceQuick:
    """Клас для швидкого збереження фічей в БД"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        
    def save_features_to_db(self):
        """Зберігає фічі в БД"""
        return quick_persist_features()


def quick_persist_features():
    """Швидке збереження базових фічей в БД"""
    
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'whale_tracker_2024',
        'database': 'gt14_whaletracker'
    }
    
    try:
        # 1. Створюємо спрощену таблицю
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        logger.info("Створення таблиці whale_features_basic...")
        
        create_table = """
        CREATE TABLE IF NOT EXISTS whale_features_basic (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            btc_price DECIMAL(20,8),
            whale_volume_usd DECIMAL(30,2),
            net_flow DECIMAL(30,2),
            whale_activity INT,
            exchange_inflow DECIMAL(30,2),
            exchange_outflow DECIMAL(30,2),
            fear_greed_index INT,
            -- Додаткові важливі фічі
            whale_tx_per_hour DECIMAL(20,8),
            whale_pressure_index DECIMAL(20,8),
            market_momentum DECIMAL(20,8),
            btc_price_lag1 DECIMAL(20,8),
            btc_price_lag24 DECIMAL(20,8),
            whale_volume_rolling24_mean DECIMAL(30,2),
            net_flow_rolling24_mean DECIMAL(30,2),
            rsi_14 DECIMAL(10,4),
            INDEX idx_timestamp (timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        
        cursor.execute(create_table)
        logger.info("Таблиця створена")
        
        # 2. Завантажуємо дані
        query = """
        SELECT * FROM whale_hourly_complete
        WHERE btc_price > 0
        ORDER BY timestamp
        LIMIT 5000
        """
        
        df = pd.read_sql(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Завантажено {len(df)} записів")
        
        # 3. Генеруємо базові фічі
        logger.info("Генерація базових фічей...")
        
        # Whale activity per hour
        df['whale_tx_per_hour'] = df['whale_activity']
        
        # Whale pressure index
        df['whale_pressure_index'] = (
            df['whale_volume_usd'] * df['whale_activity'] / 
            (df['btc_price'] + 1e-8)
        )
        
        # Market momentum
        df['market_momentum'] = (
            df['net_flow'].rolling(6, min_periods=1).mean() * 
            df['btc_price'].pct_change(6).fillna(0)
        )
        
        # Лаги
        df['btc_price_lag1'] = df['btc_price'].shift(1)
        df['btc_price_lag24'] = df['btc_price'].shift(24)
        
        # Rolling means
        df['whale_volume_rolling24_mean'] = df['whale_volume_usd'].rolling(24, min_periods=1).mean()
        df['net_flow_rolling24_mean'] = df['net_flow'].rolling(24, min_periods=1).mean()
        
        # Simple RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14'] = calculate_rsi(df['btc_price'])
        
        # 4. Зберігаємо в БД
        logger.info("Збереження в БД...")
        
        # Очищаємо таблицю
        cursor.execute("TRUNCATE TABLE whale_features_basic")
        
        # Підготовка даних
        columns = [
            'timestamp', 'btc_price', 'whale_volume_usd', 'net_flow',
            'whale_activity', 'exchange_inflow', 'exchange_outflow',
            'fear_greed_index', 'whale_tx_per_hour', 'whale_pressure_index',
            'market_momentum', 'btc_price_lag1', 'btc_price_lag24',
            'whale_volume_rolling24_mean', 'net_flow_rolling24_mean', 'rsi_14'
        ]
        
        df_to_save = df[columns].copy()
        
        # Заповнюємо пропущені значення
        df_to_save = df_to_save.fillna({
            'whale_tx_per_hour': 0,
            'whale_pressure_index': 0,
            'market_momentum': 0,
            'btc_price_lag1': df_to_save['btc_price'],
            'btc_price_lag24': df_to_save['btc_price'],
            'whale_volume_rolling24_mean': df_to_save['whale_volume_usd'],
            'net_flow_rolling24_mean': df_to_save['net_flow'],
            'rsi_14': 50
        })
        
        # Вставка даних
        insert_query = f"""
        INSERT INTO whale_features_basic 
        ({', '.join(columns)})
        VALUES ({', '.join(['%s'] * len(columns))})
        """
        
        # Конвертуємо NaN в None для MySQL
        df_to_save = df_to_save.where(pd.notnull(df_to_save), None)
        
        values = [tuple(row) for row in df_to_save.values]
        cursor.executemany(insert_query, values)
        
        conn.commit()
        logger.info(f"Збережено {len(values)} записів")
        
        # 5. Верифікація
        cursor.execute("SELECT COUNT(*) FROM whale_features_basic")
        count = cursor.fetchone()[0]
        logger.info(f"Верифікація: в таблиці {count} записів")
        
        cursor.close()
        conn.close()
        
        return count
        
    except Exception as e:
        logger.error(f"Помилка: {str(e)}")
        return 0


if __name__ == "__main__":
    count = quick_persist_features()
    if count > 0:
        print(f"\n✅ Успішно збережено {count} записів з базовими фічами!")
    else:
        print("\n❌ Помилка збереження фічей")