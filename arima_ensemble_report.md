# ARIMA Ensemble Models - Звіт

**Дата аналізу:** 2025-07-02 05:58:28

## Результати по моделях

| Модель | Тип | MAPE (%) | RMSE | MAE | AIC |
|--------|-----|----------|------|-----|-----|
| HoltWinters_add | HoltWinters | 0.00 | 3.14 | nan | 32512.20 |
| HoltWinters_mul | HoltWinters | 0.00 | 3.38 | nan | 32512.06 |
| ARIMA_101 | ARIMA | 0.02 | 8.16 | nan | 41166.40 |
| ARIMA_212 | ARIMA | 0.11 | 60.07 | nan | 38330.49 |
| ARIMA_211 | ARIMA | 0.14 | 81.07 | nan | 38332.59 |
| ARIMA_111 | ARIMA | 0.29 | 164.33 | nan | 38404.68 |
| SARIMA_111_111_24 | SARIMA | 0.31 | 175.24 | nan | 38004.13 |
| SARIMA_101_110_24 | SARIMA | 0.90 | 555.10 | nan | 42170.88 |

## Найкраща модель: HoltWinters_add
- MAPE: 0.00%

## Конфігурації моделей

### ARIMA_101
```python
{
  "order": [
    1,
    0,
    1
  ],
  "type": "ARIMA"
}
```

### ARIMA_111
```python
{
  "order": [
    1,
    1,
    1
  ],
  "type": "ARIMA"
}
```

### ARIMA_211
```python
{
  "order": [
    2,
    1,
    1
  ],
  "type": "ARIMA"
}
```

### ARIMA_212
```python
{
  "order": [
    2,
    1,
    2
  ],
  "type": "ARIMA"
}
```

### SARIMA_111_111_24
```python
{
  "order": [
    1,
    1,
    1
  ],
  "seasonal_order": [
    1,
    1,
    1,
    24
  ],
  "type": "SARIMA"
}
```

### SARIMA_101_110_24
```python
{
  "order": [
    1,
    0,
    1
  ],
  "seasonal_order": [
    1,
    1,
    0,
    24
  ],
  "type": "SARIMA"
}
```

### HoltWinters_add
```python
{
  "seasonal": "add",
  "seasonal_periods": 24,
  "type": "HoltWinters"
}
```

### HoltWinters_mul
```python
{
  "seasonal": "mul",
  "seasonal_periods": 24,
  "type": "HoltWinters"
}
```

