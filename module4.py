def mod4(all_data):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Среднее и стандартное отклонение для теоретической модели
    mean_val = np.mean(all_data)
    std_dev = np.std(all_data)

    # a) Гистограмма и её теоретический аналог
    plt.figure(figsize=(9, 6))
    # Гистограмма относительных частот
    count, bins, ignored = plt.hist(all_data, bins=10, density=True, alpha=0.5, color='g', edgecolor='black')

    # Теоретическая плотность вероятности
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_val, std_dev)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Гистограмма и её теоретический аналог плотности вероятности')
    plt.xlabel('Значение')
    plt.ylabel('Вероятность')
    plt.grid(True)
    plt.show()

    # б) Эмпирическая и теоретическая функции распределения
    # Эмпирическая функция распределения
    sorted_data = np.sort(all_data)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n + 1) / n

    plt.figure(figsize=(9, 6))
    plt.step(sorted_data, empirical_cdf, where='post', label='Эмпирическая F*(x)', color='b')

    # Теоретическая функция распределения
    theoretical_cdf = norm.cdf(x, mean_val, std_dev)
    plt.plot(x, theoretical_cdf, 'r', label='Теоретическая F(x)', linewidth=2)

    plt.title('Эмпирическая и теоретическая функции распределения')
    plt.xlabel('Значение')
    plt.ylabel('Функция распределения')
    plt.grid(True)
    plt.legend()
    plt.show()