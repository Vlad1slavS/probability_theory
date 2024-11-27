def check_sigma_rule(data):
    import numpy as np

    mean = np.mean(data)
    sigma = np.std(data)

    within_1_sigma = np.sum((data >= mean - sigma) & (data <= mean + sigma))
    within_2_sigma = np.sum((data >= mean - 2 * sigma) & (data <= mean + 2 * sigma))
    within_3_sigma = np.sum((data >= mean - 3 * sigma) & (data <= mean + 3 * sigma))

    n = len(data)
    print(f"Процент значений в пределах 1σ: {within_1_sigma / n * 100:.2f}%")
    print(f"Процент значений в пределах 2σ: {within_2_sigma / n * 100:.2f}%")
    print(f"Процент значений в пределах 3σ: {within_3_sigma / n * 100:.2f}%")