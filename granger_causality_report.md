# Granger Causality Analysis Report

**Дата аналізу:** 2025-07-02 05:57:34

## Загальна статистика
- Всього тестів: 229
- Значущих зв'язків (p<0.05): 111
- Відсоток значущих: 48.5%

## Найсильніші причинні зв'язки

| Зв'язок | F-stat | p-value | Lag | Значущий |
|---------|--------|---------|-----|----------|
| net_flow → net_flow_lag6 | 1326.0273 | 0.0000 | 5 | ✓ |
| whale_volume_ma30 → whale_volume_lag1 | 1278.7762 | 0.0000 | 2 | ✓ |
| whale_volume_ma7 → whale_volume_lag1 | 1221.2856 | 0.0000 | 2 | ✓ |
| whale_volume_usd → whale_volume_lag6 | 775.5708 | 0.0000 | 5 | ✓ |
| btc_price → target_price | 749.3610 | 0.0000 | 1 | ✓ |
| whale_volume_ma30 → whale_volume_lag3 | 627.9479 | 0.0000 | 4 | ✓ |
| whale_volume_ma7 → whale_volume_lag3 | 597.7572 | 0.0000 | 4 | ✓ |
| exchange_outflow → net_flow_lag1 | 593.8331 | 0.0000 | 2 | ✓ |
| whale_volume_ma7 → whale_volume_usd | 360.1895 | 0.0000 | 1 | ✓ |
| whale_volume_ma30 → whale_volume_usd | 340.9562 | 0.0000 | 1 | ✓ |
| net_flow_lag1 → net_flow | 300.9001 | 0.0000 | 1 | ✓ |
| whale_volume_lag1 → whale_volume_usd | 299.1032 | 0.0000 | 1 | ✓ |
| exchange_inflow → net_flow_lag1 | 241.3477 | 0.0000 | 4 | ✓ |
| exchange_outflow → net_flow_lag3 | 233.8600 | 0.0000 | 5 | ✓ |
| exchange_inflow → net_flow_lag3 | 196.9375 | 0.0000 | 5 | ✓ |

## Аналіз по змінних

### Змінні з найбільшим впливом (causes):
- **exchange_inflow**: впливає на 12 змінних
- **fear_greed_index**: впливає на 10 змінних
- **net_flow**: впливає на 9 змінних
- **whale_activity**: впливає на 9 змінних
- **exchange_outflow**: впливає на 9 змінних
- **whale_volume_ma7**: впливає на 9 змінних
- **whale_volume_ma30**: впливає на 9 змінних
- **whale_volume_lag1**: впливає на 8 змінних
- **net_flow_lag1**: впливає на 8 змінних
- **net_flow_lag3**: впливає на 7 змінних
- **whale_volume_usd**: впливає на 5 змінних
- **whale_volume_lag3**: впливає на 5 змінних
- **whale_volume_lag6**: впливає на 4 змінних
- **btc_price**: впливає на 3 змінних
- **net_flow_lag6**: впливає на 3 змінних
- **target_price**: впливає на 1 змінних

### Змінні з найбільшою залежністю (effects):
- **whale_volume_usd**: залежить від 11 змінних
- **whale_volume_lag1**: залежить від 11 змінних
- **whale_volume_lag3**: залежить від 10 змінних
- **exchange_inflow**: залежить від 9 змінних
- **exchange_outflow**: залежить від 9 змінних
- **whale_volume_ma30**: залежить від 8 змінних
- **whale_volume_lag6**: залежить від 8 змінних
- **net_flow_lag6**: залежить від 8 змінних
- **net_flow_lag3**: залежить від 7 змінних
- **whale_activity**: залежить від 7 змінних
- **whale_volume_ma7**: залежить від 6 змінних
- **net_flow_lag1**: залежить від 6 змінних
- **fear_greed_index**: залежить від 5 змінних
- **net_flow**: залежить від 4 змінних
- **target_price**: залежить від 1 змінних
- **btc_price**: залежить від 1 змінних

## Рекомендації
1. Використовувати змінні з сильними причинними зв'язками для прогнозування
2. Враховувати оптимальні лаги при побудові моделей
3. Звернути увагу на двонаправлені зв'язки
