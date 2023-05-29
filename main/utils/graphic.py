"""
Here diograms with logged information about prices can be visualized

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

k = 50 # сколько объектов исаользовать

rp_f = open("logs/real_price_bernulli.txt")
real_prices = list(map(lambda x: float(x), rp_f.readlines()[:k]))

fp_e_f = open("logs/final_price_greedy.txt")
final_prices_greedy = list(map(lambda x: float(x), fp_e_f.readlines()[:k]))

fp_b_f = open("logs/final_price_bernulli.txt")
final_prices_b = list(map(lambda x: float(x), fp_b_f.readlines()[:k]))

fp_gauss_f = open("logs/final_price_gauss.txt")
final_prices_gauss = list(map(lambda x: float(x), fp_gauss_f.readlines()[:k]))

plt.plot(real_prices, label="real_prices")
plt.plot(final_prices_greedy, label="e-greedy")
plt.plot(final_prices_b, label="bernulli")
plt.plot(final_prices_gauss, label="gauss")
plt.xlabel("Объект")
plt.ylabel("Цена")
plt.legend()
plt.savefig('pictures/price_cmp.png')
plt.show()
