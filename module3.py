import numpy as np
from scipy import stats

def mod3 (all_data):
    mode_val = stats.mode(all_data)[0]

    # Медиана
    median_val = np.median(all_data)

    # Среднее арифметическое
    mean_val = np.mean(all_data)

    # Стандартное отклонение

    std_dev = np.std(all_data)

    # Вывод результатов
    print(f"Мода (М°): {mode_val}")
    print(f"Медиана (М*): {median_val}")
    print(f"Среднее арифметическое (*): {mean_val}")
    print(f"Стандартное отклонение (σ): {std_dev}")

    return mean_val, std_dev