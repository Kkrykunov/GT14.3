#!/usr/bin/env python3
"""
ЕТАП 2: Універсальна система генерації фічей без контексту об'єкта
Створює фічі які працюють з будь-яким timeline
"""

import pandas as pd
import numpy as np
import mysql.connector
from sklearn.preprocessing import StandardScaler, RobustScaler
import ta  # технічні індикатори
import warnings
warnings.filterwarnings('ignore')

class UniversalFeatureEngine:
    """Генератор універсальних фічей без контексту об'єкта"""
    
    def __init__(self):
        self.config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'whale_tracker_2024',
            'database': 'gt14_whaletracker'
        }
        
        self.generated_features = []
        self.feature_importance = {}
        
    def generate_universal_features(self, df=None, target_col='btc_price'):
        """Генерація універсальних фічей для будь-якого timeline"""
        
        print(" ЕТАП 2: УНІВЕРСАЛЬНА ГЕНЕРАЦІЯ ФІЧЕЙ")
        print("=" * 60)
        
        # Завантаження даних якщо не передано
        if df is None:
            df = self.load_data()
        
        print(f" Вхідні дані: {df.shape}")
        print(f" Цільова змінна: {target_col}")
        
        # Очищаємо список згенерованих фічей
        self.generated_features = []
        
        # 1. ЧАСОВІ ФІЧІ (без контексту)
        df_with_features = self.add_temporal_features(df.copy())
        
        # 2. ЛАГОВІ ФІЧІ 
        df_with_features = self.add_lag_features(df_with_features)
        
        # 3. ROLLING STATISTICS ФІЧІ
        df_with_features = self.add_rolling_features(df_with_features)
        
        # 4. ТЕХНІЧНІ ІНДИКАТОРИ ФІЧІ
        df_with_features = self.add_technical_features(df_with_features)
        
        # 5. WHALE-СПЕЦИФІЧНІ ФІЧІ
        df_with_features = self.add_whale_features(df_with_features)
        
        # 6. ВОЛАТИЛЬНІСТЬ ФІЧІ
        df_with_features = self.add_volatility_features(df_with_features)
        
        # 7. ВЗАЄМОДІЇ ФІЧЕЙ (FEATURE INTERACTIONS)
        df_with_features = self.add_interaction_features(df_with_features)
        
        # 8. СТАТИСТИЧНІ ТРАНСФОРМАЦІЇ
        df_with_features = self.add_statistical_transforms(df_with_features)
        
        # 9. ЗБЕРІГАЄМО В MYSQL
        self.save_features_to_mysql(df_with_features)
        
        print(f"\n ГЕНЕРАЦІЯ ЗАВЕРШЕНА:")
        print(f"   Початкових колонок: {df.shape[1]}")
        print(f"   Згенерованих фічей: {len(self.generated_features)}")
        print(f"   Загальних колонок: {df_with_features.shape[1]}")
        
        return df_with_features
    
    def load_data(self):
        """Завантаження даних з MySQL"""
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
            fear_greed_index
        FROM whale_hourly_complete
        WHERE whale_activity > 0
        AND btc_price > 0
        ORDER BY timestamp
        """
        
        df = pd.read_sql(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        conn.close()
        
        return df
    
    def add_temporal_features(self, df):
        """Часові фічі без контексту"""
        print("⏰ Генерація часових фічей...")
        
        # Години доби
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Дні тижня
        df['dayofweek'] = df.index.dayofweek
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Дні місяця
        df['dayofmonth'] = df.index.day
        df['dayofmonth_sin'] = np.sin(2 * np.pi * df['dayofmonth'] / 31)
        df['dayofmonth_cos'] = np.cos(2 * np.pi * df['dayofmonth'] / 31)
        
        # Місяці року
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Вихідні
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Час від початку даних (тренд)
        df['time_trend'] = range(len(df))
        
        temporal_features = [
            'hour', 'hour_sin', 'hour_cos',
            'dayofweek', 'dayofweek_sin', 'dayofweek_cos',
            'dayofmonth', 'dayofmonth_sin', 'dayofmonth_cos',
            'month', 'month_sin', 'month_cos',
            'is_weekend', 'time_trend'
        ]
        
        self.generated_features.extend(temporal_features)
        print(f"    Додано {len(temporal_features)} часових фічей")
        
        return df
    
    def add_lag_features(self, df):
        """Лагові фічі"""
        print(" Генерація лагових фічей...")
        
        base_columns = ['whale_volume_usd', 'exchange_inflow', 'exchange_outflow', 'net_flow', 'btc_price']
        lags = [1, 3, 6, 12, 24, 48]  # 1h, 3h, 6h, 12h, 24h, 48h
        
        lag_features = []
        
        for col in base_columns:
            if col in df.columns:
                for lag in lags:
                    feature_name = f'{col}_lag{lag}'
                    df[feature_name] = df[col].shift(lag)
                    lag_features.append(feature_name)
        
        self.generated_features.extend(lag_features)
        print(f"    Додано {len(lag_features)} лагових фічей")
        
        return df
    
    def add_rolling_features(self, df):
        """Rolling statistics фічі"""
        print(" Генерація rolling statistics фічей...")
        
        base_columns = ['whale_volume_usd', 'exchange_inflow', 'exchange_outflow', 'net_flow', 'btc_price']
        windows = [3, 6, 12, 24, 48, 168]  # 3h, 6h, 12h, 24h, 48h, 7d
        stats = ['mean', 'std', 'min', 'max', 'median']
        
        rolling_features = []
        
        for col in base_columns:
            if col in df.columns:
                for window in windows:
                    for stat in stats:
                        feature_name = f'{col}_rolling{window}_{stat}'
                        
                        if stat == 'median':
                            df[feature_name] = df[col].rolling(window).median()
                        else:
                            df[feature_name] = getattr(df[col].rolling(window), stat)()
                        
                        rolling_features.append(feature_name)
        
        self.generated_features.extend(rolling_features)
        print(f"    Додано {len(rolling_features)} rolling statistics фічей")
        
        return df
    
    def add_technical_features(self, df):
        """Технічні індикатори фічі"""
        print(" Генерація технічних індикаторів...")
        
        if 'btc_price' not in df.columns:
            print("    btc_price не знайдено, пропускаємо технічні індикатори")
            return df
        
        technical_features = []
        
        # RSI
        for period in [14, 30]:
            feature_name = f'rsi_{period}'
            df[feature_name] = ta.momentum.RSIIndicator(df['btc_price'], window=period).rsi()
            technical_features.append(feature_name)
        
        # MACD
        macd = ta.trend.MACD(df['btc_price'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        technical_features.extend(['macd', 'macd_signal', 'macd_diff'])
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['btc_price'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = df['bb_high'] - df['bb_low']
        df['bb_position'] = (df['btc_price'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        technical_features.extend(['bb_high', 'bb_low', 'bb_mid', 'bb_width', 'bb_position'])
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['btc_price'], df['btc_price'], df['btc_price'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        technical_features.extend(['stoch_k', 'stoch_d'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['btc_price'], df['btc_price'], df['btc_price']
        ).williams_r()
        technical_features.append('williams_r')
        
        self.generated_features.extend(technical_features)
        print(f"    Додано {len(technical_features)} технічних індикаторів")
        
        return df
    
    def add_whale_features(self, df):
        """Whale-специфічні фічі"""
        print(" Генерація whale-специфічних фічей...")
        
        whale_features = []
        
        # Інтенсивність whale активності
        if 'whale_activity' in df.columns:
            df['whale_intensity'] = df['whale_activity'].rolling(24).sum() / 24
            whale_features.append('whale_intensity')
        
        # Flow ratios
        if 'exchange_inflow' in df.columns and 'exchange_outflow' in df.columns:
            df['flow_ratio'] = df['exchange_outflow'] / (df['exchange_inflow'] + 1)
            df['flow_imbalance'] = (df['exchange_outflow'] - df['exchange_inflow']) / (df['exchange_outflow'] + df['exchange_inflow'] + 1)
            whale_features.extend(['flow_ratio', 'flow_imbalance'])
        
        # Whale volume ratios
        if 'whale_volume_usd' in df.columns:
            df['whale_volume_normalized'] = df['whale_volume_usd'] / df['whale_volume_usd'].rolling(168).mean()
            df['whale_volume_percentile'] = df['whale_volume_usd'].rolling(168).rank(pct=True)
            whale_features.extend(['whale_volume_normalized', 'whale_volume_percentile'])
        
        # Large transaction detection
        if 'whale_volume_usd' in df.columns:
            threshold_95 = df['whale_volume_usd'].quantile(0.95)
            threshold_99 = df['whale_volume_usd'].quantile(0.99)
            
            df['is_large_tx'] = (df['whale_volume_usd'] > threshold_95).astype(int)
            df['is_mega_tx'] = (df['whale_volume_usd'] > threshold_99).astype(int)
            df['large_tx_frequency'] = df['is_large_tx'].rolling(24).sum()
            
            whale_features.extend(['is_large_tx', 'is_mega_tx', 'large_tx_frequency'])
        
        # Enhanced whale activity intensity (транзакцій на годину)
        if 'whale_activity' in df.columns:
            # Кількість транзакцій на годину з різними вікнами
            df['whale_tx_per_hour'] = df['whale_activity']
            df['whale_tx_per_hour_3h'] = df['whale_activity'].rolling(3).mean()
            df['whale_tx_per_hour_24h'] = df['whale_activity'].rolling(24).mean()
            whale_features.extend(['whale_tx_per_hour', 'whale_tx_per_hour_3h', 'whale_tx_per_hour_24h'])
        
        # Cross-exchange flow patterns
        if 'exchange_inflow' in df.columns and 'exchange_outflow' in df.columns:
            # Патерни потоків між біржами
            df['flow_momentum'] = df['net_flow'].rolling(6).mean()
            df['flow_acceleration'] = df['flow_momentum'].diff()
            
            # Кореляція між inflow та outflow (зсув на 1-3 години)
            df['flow_correlation_1h'] = df['exchange_inflow'].rolling(24).corr(df['exchange_outflow'].shift(1))
            df['flow_correlation_3h'] = df['exchange_inflow'].rolling(24).corr(df['exchange_outflow'].shift(3))
            
            # Виявлення аномальних flow патернів
            flow_mean = df['net_flow'].rolling(168).mean()
            flow_std = df['net_flow'].rolling(168).std()
            df['flow_zscore'] = (df['net_flow'] - flow_mean) / (flow_std + 1e-8)
            
            whale_features.extend(['flow_momentum', 'flow_acceleration', 
                                 'flow_correlation_1h', 'flow_correlation_3h', 'flow_zscore'])
        
        self.generated_features.extend(whale_features)
        print(f"    Додано {len(whale_features)} whale-специфічних фічей")
        
        return df
    
    def add_volatility_features(self, df):
        """Волатільність фічі"""
        print(" Генерація волатільності фічей...")
        
        if 'btc_price' not in df.columns:
            return df
        
        volatility_features = []
        
        # Price changes
        df['price_change_1h'] = df['btc_price'].pct_change(1)
        df['price_change_24h'] = df['btc_price'].pct_change(24)
        volatility_features.extend(['price_change_1h', 'price_change_24h'])
        
        # Realized volatility
        for window in [24, 168]:  # 1 day, 1 week
            feature_name = f'realized_vol_{window}h'
            df[feature_name] = df['price_change_1h'].rolling(window).std() * np.sqrt(window)
            volatility_features.append(feature_name)
        
        # Parkinson volatility estimator
        if 'btc_price' in df.columns:
            df['high_low_ratio'] = df['btc_price'] / df['btc_price'].rolling(24).min()
            volatility_features.append('high_low_ratio')
        
        self.generated_features.extend(volatility_features)
        print(f"    Додано {len(volatility_features)} волатільності фічей")
        
        return df
    
    def add_interaction_features(self, df):
        """Взаємодії фічей"""
        print(" Генерація взаємодій фічей...")
        
        interaction_features = []
        
        # Whale volume * Fear & Greed
        if 'whale_volume_usd' in df.columns and 'fear_greed_index' in df.columns:
            df['whale_fear_interaction'] = df['whale_volume_usd'] * df['fear_greed_index']
            interaction_features.append('whale_fear_interaction')
        
        # Flow * Price change
        if 'net_flow' in df.columns and 'price_change_1h' in df.columns:
            df['flow_price_interaction'] = df['net_flow'] * df['price_change_1h']
            interaction_features.append('flow_price_interaction')
        
        # Activity * Volatility
        if 'whale_activity' in df.columns and 'realized_vol_24h' in df.columns:
            df['activity_vol_interaction'] = df['whale_activity'] * df['realized_vol_24h']
            interaction_features.append('activity_vol_interaction')
        
        self.generated_features.extend(interaction_features)
        print(f"    Додано {len(interaction_features)} взаємодій фічей")
        
        return df
    
    def add_statistical_transforms(self, df):
        """Статистичні трансформації"""
        print(" Генерація статистичних трансформацій...")
        
        transform_features = []
        base_columns = ['whale_volume_usd', 'exchange_inflow', 'exchange_outflow']
        
        for col in base_columns:
            if col in df.columns:
                # Log transform
                df[f'{col}_log'] = np.log1p(df[col])
                transform_features.append(f'{col}_log')
                
                # Square root transform
                df[f'{col}_sqrt'] = np.sqrt(df[col])
                transform_features.append(f'{col}_sqrt')
                
                # Z-score (rolling)
                rolling_mean = df[col].rolling(168).mean()
                rolling_std = df[col].rolling(168).std()
                df[f'{col}_zscore'] = (df[col] - rolling_mean) / rolling_std
                transform_features.append(f'{col}_zscore')
        
        self.generated_features.extend(transform_features)
        print(f"    Додано {len(transform_features)} статистичних трансформацій")
        
        return df
    
    def save_features_to_mysql(self, df):
        """Збереження фічей в MySQL"""
        print("\n ЗБЕРЕЖЕННЯ ФІЧЕЙ В MYSQL...")
        
        conn = mysql.connector.connect(**self.config)
        cursor = conn.cursor()
        
        # Створюємо таблицю для універсальних фічей
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS universal_features (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            feature_name VARCHAR(100),
            feature_value DECIMAL(20,6),
            feature_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_timestamp (timestamp),
            INDEX idx_feature_name (feature_name)
        )
        """)
        
        # Очищуємо стару таблицю
        cursor.execute("DELETE FROM universal_features")
        
        # Підготовка даних для вставки
        insert_data = []
        
        for feature in self.generated_features:
            if feature in df.columns:
                feature_type = self.get_feature_type(feature)
                
                for timestamp, value in df[feature].items():
                    if pd.notna(value):
                        insert_data.append((
                            timestamp,
                            feature,
                            float(value),
                            feature_type
                        ))
        
        # Вставка даних у батчах
        if insert_data:
            batch_size = 1000  # Зменшуємо розмір батчу
            for i in range(0, len(insert_data), batch_size):
                batch = insert_data[i:i + batch_size]
                cursor.executemany("""
                INSERT INTO universal_features 
                (timestamp, feature_name, feature_value, feature_type)
                VALUES (%s, %s, %s, %s)
                """, batch)
                print(f"   Збережено батч {i//batch_size + 1}: {len(batch)} записів")
        
        # Створюємо метадані фічей
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_metadata (
            feature_name VARCHAR(100) PRIMARY KEY,
            feature_type VARCHAR(50),
            description TEXT,
            importance_score DECIMAL(10,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("DELETE FROM feature_metadata")
        
        # Вставка метаданих
        metadata = []
        for feature in self.generated_features:
            metadata.append((
                feature,
                self.get_feature_type(feature),
                self.get_feature_description(feature),
                0.0  # importance буде обчислено пізніше
            ))
        
        cursor.executemany("""
        INSERT INTO feature_metadata 
        (feature_name, feature_type, description, importance_score)
        VALUES (%s, %s, %s, %s)
        """, metadata)
        
        conn.commit()
        conn.close()
        
        print(f" Збережено {len(insert_data)} записів фічей")
        print(f" Збережено {len(metadata)} метаданих фічей")
    
    def get_feature_type(self, feature_name):
        """Визначення типу фічі"""
        if any(x in feature_name for x in ['hour', 'day', 'month', 'weekend', 'time']):
            return 'temporal'
        elif 'lag' in feature_name:
            return 'lag'
        elif 'rolling' in feature_name:
            return 'rolling_stat'
        elif any(x in feature_name for x in ['rsi', 'macd', 'bb_', 'stoch', 'williams']):
            return 'technical'
        elif any(x in feature_name for x in ['whale', 'flow', 'tx']):
            return 'whale_specific'
        elif any(x in feature_name for x in ['vol', 'change']):
            return 'volatility'
        elif 'interaction' in feature_name:
            return 'interaction'
        elif any(x in feature_name for x in ['log', 'sqrt', 'zscore']):
            return 'transform'
        else:
            return 'other'
    
    def get_feature_description(self, feature_name):
        """Опис фічі"""
        descriptions = {
            'temporal': 'Часова фіча без контексту об\'єкта',
            'lag': 'Лагова фіча для виявлення затримок',
            'rolling_stat': 'Rolling статистика для тренду',
            'technical': 'Технічний індикатор',
            'whale_specific': 'Специфічна whale метрика',
            'volatility': 'Метрика волатільності',
            'interaction': 'Взаємодія між фічами',
            'transform': 'Статистична трансформація'
        }
        feature_type = self.get_feature_type(feature_name)
        return descriptions.get(feature_type, 'Універсальна фіча')

def main():
    """Головна функція"""
    feature_engine = UniversalFeatureEngine()
    
    # Генерація фічей
    df_with_features = feature_engine.generate_universal_features()
    
    print(f"\n ЕТАП 2 ЗАВЕРШЕНО!")
    print(f"Згенеровано {len(feature_engine.generated_features)} універсальних фічей")
    print(f"Розподіл по типах:")
    
    # Статистика по типах фічей
    feature_types = {}
    for feature in feature_engine.generated_features:
        feature_type = feature_engine.get_feature_type(feature)
        feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
    
    for feature_type, count in sorted(feature_types.items()):
        print(f"  {feature_type}: {count} фічей")

if __name__ == "__main__":
    main()