import module3, module4, module6
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Данные
A = [
    [60, 71, 62, 57, 81, 55, 59, 47, 75, 56, 61, 60, 63, 65, 59, 61, 65, 58, 76, 49],
    [65, 64, 59, 76, 58, 52, 70, 77, 67, 50, 65, 53, 56, 64, 55, 77, 51, 61, 73, 64],
    [45, 53, 45, 58, 57, 60, 48, 71, 33, 65, 50, 80, 58, 67, 71, 51, 51, 49, 66, 63],
    [67, 60, 67, 61, 58, 36, 75, 47, 68, 63, 77, 75, 62, 75, 70, 75, 66, 53, 63, 60],
    [68, 67, 55, 75, 71, 59, 77, 58, 65, 57, 55, 28, 74, 71, 47, 73, 40, 45, 37, 66]
]

# Объединение всех данных в один список
all_data = np.concatenate(A)

# Найти минимальное и максимальное значения
min_val = int(np.min(all_data))
max_val = int(np.max(all_data))

# Определяем количество интервалов
num_intervals = 10

# Вычисляем размер шага для интервалов
step_size = (max_val - min_val) // num_intervals + 1

# Создаем границы интервалов
intervals = list(range(min_val, max_val + step_size, step_size))

# Подсчет количества значений в каждом интервале
hist, bin_edges = np.histogram(all_data, bins=intervals)

# Вывод результатов
for i in range(len(hist)):
    print(f"Интервал {bin_edges[i]} - {bin_edges[i+1]-1}: {hist[i]} значений")

################ 2 ##################

# Вычисление относительных частот
rel_freq = hist / len(all_data)

# a) Полигон относительных частот
midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
plt.figure(figsize=(9, 6))
plt.plot(midpoints, rel_freq, marker='o', linestyle='-')
plt.title('Полигон относительных частот')
plt.xlabel('Интервалы')
plt.ylabel('Относительная частота')
plt.grid(True)
plt.show()

# б) Гистограмма относительных частот
plt.figure(figsize=(9, 6))
plt.bar(midpoints, rel_freq, width=step_size, align='center', edgecolor='black')
plt.title('Гистограмма относительных частот')
plt.xlabel('Интервалы')
plt.ylabel('Относительная частота')
plt.grid(True)
plt.show()

# в) График эмпирической функции распределения
cumulative_freq = np.cumsum(rel_freq)
plt.figure(figsize=(9, 6))
plt.step(bin_edges[:-1], cumulative_freq, where='post', label='Эмпирическая функция распределения')
plt.title('График эмпирической функции распределения')
plt.xlabel('Значение')
plt.ylabel('Накопленная относительная частота')
plt.grid(True)
plt.show()

################ 3 ##################
module3.mod3(all_data)

################ 4 ##################
module4.mod4(all_data)

################ 5 ##################
module6.check_sigma_rule(all_data)

################ 7 ##################

# 7) Критерий согласия Пирсона

# Параметры выборки
N = len(all_data)
mean = np.mean(all_data)
std = np.std(all_data, ddof=1)  # несмещенная оценка

print(f"Среднее выборочное: {mean:.2f}")
print(f"Выборочное стандартное отклонение: {std:.2f}")

# Вычисление ожидаемых частот
expected_freq = []
from scipy.stats import norm

# Функция распределения нормального распределения
cdf = norm.cdf

# Проверка и объединение интервалов с ожидаемой частотой менее 5
# Сначала вычислим ожидаемые частоты
for i in range(len(bin_edges)-1):
    # Вероятность попасть в интервал при нормальном распределении
    p = cdf((bin_edges[i+1] - mean)/std) - cdf((bin_edges[i] - mean)/std)
    expected_freq.append(p * N)

# Проверяем ожидаемые частоты
print("\nОжидаемые частоты в интервалах:")
for i, ef in enumerate(expected_freq):
    print(f"Интервал {bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}: {ef:.2f}")

# Объединяем интервалы, где ожидаемые частоты менее 5
combined_obs = []
combined_exp = []
combined_intervals = []
i = 0
while i < len(hist):
    obs = hist[i]
    exp = expected_freq[i]
    interval_start = bin_edges[i]
    interval_end = bin_edges[i+1]
    while exp < 5 and i+1 < len(hist):
        i += 1
        obs += hist[i]
        exp += expected_freq[i]
        interval_end = bin_edges[i+1]
    combined_obs.append(obs)
    combined_exp.append(exp)
    combined_intervals.append((interval_start, interval_end))
    i += 1

# Вывод объединенных интервалов и частот
print("\nОбъединенные интервалы и их частоты:")
for i in range(len(combined_obs)):
    print(f"Интервал {combined_intervals[i][0]:.1f} - {combined_intervals[i][1]:.1f}: наблюд. частота = {combined_obs[i]}, ожидаемая частота = {combined_exp[i]:.2f}")

# Вычисление статистики критерия Пирсона
chi_square_stat = sum(((o - e) ** 2) / e for o, e in zip(combined_obs, combined_exp))
print(f"\nВычисленная статистика χ²: {chi_square_stat:.4f}")

# Число степеней свободы:
# df = кол-во интервалов после объединения - число оцениваемых параметров (2) - 1
degrees_of_freedom = len(combined_obs) - 3
print(f"Число степеней свободы: {degrees_of_freedom}")

# Критическое значение χ² для уровня значимости α = 0.025
alpha = 0.025
chi_square_critical = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
print(f"Критическое значение χ² для α={alpha}: {chi_square_critical:.4f}")

# Сравнение статистики с критическим значением
if chi_square_stat < chi_square_critical:
    print("Гипотеза о нормальном распределении принимается.")
else:
    print("Гипотеза о нормальном распределении отвергается.")

################ 8 ##################

# 8a) Полигон относительных частот и кривая нормального распределения

# Теоретические относительные частоты
x_values = np.linspace(min_val, max_val, 100)
theoretical_pdf = stats.norm.pdf(x_values, mean, std)

plt.figure(figsize=(9, 6))
# Полигон относительных частот
plt.plot(midpoints, rel_freq, marker='o', linestyle='-', label='Полигон относительных частот')
# Кривая нормального распределения
plt.plot(x_values, theoretical_pdf * step_size, label='Кривая нормального распределения')
plt.title('Полигон относительных частот и кривая нормального распределения')
plt.xlabel('Интервалы')
plt.ylabel('Частота')
plt.legend()
plt.grid(True)
plt.show()

# 8b) Гистограмма теоретических вероятностей и график f(x)

# Теоретические вероятности для каждого интервала
theoretical_probs = []
for i in range(len(bin_edges)-1):
    p = cdf((bin_edges[i+1] - mean)/std) - cdf((bin_edges[i] - mean)/std)
    theoretical_probs.append(p)

# Гистограмма теоретических вероятностей
plt.figure(figsize=(9, 6))
plt.bar(midpoints, theoretical_probs, width=step_size, align='center', edgecolor='black', label='Теоретические вероятности')

# График функции плотности нормального распределения
plt.plot(x_values, theoretical_pdf, color='red', label='Функция плотности f(x)')
plt.title('Гистограмма теоретических вероятностей и график f(x)')
plt.xlabel('Интервалы')
plt.ylabel('Вероятность / Плотность')
plt.legend()
plt.grid(True)
plt.show()

################ 9 ##################

# 9) Доверительные интервалы для генеральной средней и стандартного отклонения

gamma = 0.95
alpha = 1 - gamma

# Для среднего
t_alpha_2 = stats.t.ppf(1 - alpha/2, N-1)
mean_std_error = std / np.sqrt(N)
mean_ci_lower = mean - t_alpha_2 * mean_std_error
mean_ci_upper = mean + t_alpha_2 * mean_std_error
print(f"\nДоверительный интервал для генерального среднего при γ={gamma}:")
print(f"({mean_ci_lower:.2f}; {mean_ci_upper:.2f})")

# Для стандартного отклонения
chi2_lower = stats.chi2.ppf(alpha/2, N-1)
chi2_upper = stats.chi2.ppf(1 - alpha/2, N-1)
sigma_ci_lower = std * np.sqrt((N - 1) / chi2_upper)
sigma_ci_upper = std * np.sqrt((N - 1) / chi2_lower)
print(f"\nДоверительный интервал для генерального стандартного отклонения при γ={gamma}:")
print(f"({sigma_ci_lower:.2f}; {sigma_ci_upper:.2f})")
