import numpy as np
import matplotlib.pyplot as plt

# p의 값 범위 설정
p = np.linspace(-2, 2, 400)

# 주어진 식
lambda_p = 2 / (1 + np.exp(-10 * p)) - 1

# 그래프 그리기
plt.plot(p, lambda_p)
plt.title(r'$\lambda_p = \frac{2}{1 + e^{-10p}} - 1$')
plt.xlabel('p')
plt.ylabel(r'$\lambda_p$')
plt.grid(True)
plt.show()
