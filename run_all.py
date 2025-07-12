#!/usr/bin/env python3
"""
Скрипт для запуску всіх компонентів GT14 v14.2
"""

import subprocess
import sys
import time
from datetime import datetime

def run_script(script_name, description):
    """Запуск скрипта з виводом статусу"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"Запуск: {script_name}")
    print(f"Час: {datetime.now()}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f" {description} - ЗАВЕРШЕНО УСПІШНО")
        else:
            print(f" {description} - ПОМИЛКА")
            return False
    except Exception as e:
        print(f" Помилка запуску {script_name}: {e}")
        return False
    
    return True

def main():
    """Головна функція"""
    print("""
    
              GT14 WhaleTracker v14.2 - ПОВНИЙ ЗАПУСК        
    
    """)
    
    scripts = [
        ("GT14_v14_2_COMPLETE_ENHANCED_PIPELINE.py", "ОСНОВНИЙ PIPELINE"),
        ("cluster_detailed_analysis.py", "ДЕТАЛЬНИЙ АНАЛІЗ КЛАСТЕРІВ"),
        ("interactive_clustering.py", "ІНТЕРАКТИВНА КЛАСТЕРИЗАЦІЯ"),
        ("universal_feature_engineering.py", "ГЕНЕРАЦІЯ УНІВЕРСАЛЬНИХ ФІЧЕЙ"),
        ("self_learning_arima.py", "САМОНАВЧАЛЬНА ARIMA МОДЕЛЬ")
    ]
    
    print("Буде запущено наступні скрипти:")
    for i, (script, desc) in enumerate(scripts, 1):
        print(f"  {i}. {desc}")
    
    choice = input("\nВиберіть опцію:\n1. Запустити всі\n2. Запустити вибірково\n3. Вихід\n\nВаш вибір: ")
    
    if choice == '1':
        # Запуск всіх скриптів
        success_count = 0
        for script, description in scripts:
            if run_script(script, description):
                success_count += 1
            time.sleep(2)  # Пауза між скриптами
        
        print(f"\n{'='*60}")
        print(f"ПІДСУМОК: Успішно виконано {success_count}/{len(scripts)} скриптів")
        print('='*60)
        
    elif choice == '2':
        # Вибірковий запуск
        print("\nВведіть номери скриптів через кому (наприклад: 1,3,5):")
        selected = input().strip().split(',')
        
        success_count = 0
        total_selected = 0
        
        for idx in selected:
            try:
                i = int(idx.strip()) - 1
                if 0 <= i < len(scripts):
                    script, description = scripts[i]
                    if run_script(script, description):
                        success_count += 1
                    total_selected += 1
                    time.sleep(2)
            except:
                print(f" Невірний номер: {idx}")
        
        print(f"\n{'='*60}")
        print(f"ПІДСУМОК: Успішно виконано {success_count}/{total_selected} скриптів")
        print('='*60)
    
    else:
        print("Вихід...")

if __name__ == "__main__":
    main()