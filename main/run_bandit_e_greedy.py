data_path = "main/data/all_ready_data.csv"
start_index = 20000 #начала среза данных
m = 1000  #- количество ручек
n = 10000 # количество квартир
start_dispersion = 500000 # - для генерации ручек
selling_treshold = 30 # - количество просмотров, на которое приходится одна покупка
T = 1000 # кол-во шагов

epsilon = 0.3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import helpers.demand_class as demand
import helpers.start_price_class as start_price
from scipy.stats import bernoulli
import time

start_time = time.time()


data = pd.read_csv(data_path)[start_index:start_index+n]
data_y  = data['Цена'].values

f = open("main/logs/real_price_greedy.txt", "w")
for p in list(data_y):
    print(p, file=f)


df_num = data.select_dtypes(exclude=['object'])
df_obj = data.select_dtypes(include=['object']).copy()
for c in df_obj:
    print("Warning: some not factorized field:", c)
    df_obj[c] = pd.factorize(df_obj[c])[0]
data = pd.concat([df_num, df_obj], axis=1)




start_price_model = start_price.StartPriceModel("main/models/weights/start_price.model")
demand_model = demand.DemandModel("main/models/weights/demand.model")




def get_real_profit(p, data_t):
    d2 = data_t.copy()
    d2["Цена"] = p
    d2.drop(['view_av'], axis=1, inplace=True)
    preds = np.clip(demand_model.predict(d2), 0, 5000)
    f = np.round(preds/selling_treshold) 
    R = f * p
    return R, np.sum(f)


# создаем ручки (цены) для всех квартир

df = data.copy()
df.drop(['Цена'], axis=1,inplace=True)
df.drop(['view_av'], axis=1,inplace=True)
# print(df.dtypes)
start_prices = start_price_model.predict(df)


results = list()
selled = list()
R, s = get_real_profit(start_prices, data)
results.append(sum(R))
selled.append(s)


all_prices = list()
for i in range(n):
    prices_one = np.random.normal(start_prices[i], start_dispersion, m)
    all_prices.append(prices_one)


mean_rewards = np.zeros((n, m))
num_pulling = np.zeros((n, m))
curr_prices = start_prices
print(curr_prices)
start_cycle_time = time.time()
for t in range(T):
    print(t, "/", T)
    print("Продано:", s)
    curr_prices = list()
    best_arms = list()
    for i in range(n): # для всех квартир
        coin_result = bernoulli.rvs(p=epsilon)
        if coin_result == 1:
            arm = np.random.choice([j for j in range(m)])
            best_arms.append(arm)
            curr_prices.append(all_prices[i][arm])
            num_pulling[i][arm] += 1
        else:
            best_arm = mean_rewards[i].argmax()
            arm = best_arm
            best_arms.append(arm)
            curr_prices.append(all_prices[i][arm])
            num_pulling[i][arm] += 1
    R, s = get_real_profit(curr_prices, data)
    results.append(sum(R))
    selled.append(s)
    for i in range(n):
        if num_pulling[i][best_arms[i]] != 0:
            mean_rewards[i][best_arms[i]] = (mean_rewards[i][[best_arms[i]]] * (num_pulling[i][[best_arms[i]]] - 1) + R[i])/num_pulling[i][best_arms[i]]

end_time = time.time()

f2 = open("main/logs/final_price_greedy.txt", "w")
for p in curr_prices:
    print(p, file=f2)


print("MAE = ", mean_absolute_error(data_y, curr_prices))
print("MSE = ", mean_squared_error(data_y, curr_prices))
print("MAPE = ", 100 * mean_absolute_percentage_error(data_y, curr_prices))
print("ME = ", sum(curr_prices-data_y)/len(data_y))

print("Среднее время исполнения шага:", (end_time - start_cycle_time)/T)
print("Время выполнения", end_time - start_time)
print("Прибыль", results[-1])
plt.plot(results)
plt.xlabel("Количество шагов")
plt.ylabel("Прибыль")
plt.savefig('main/pictures/results_e_greedy.png')
plt.show()
plt.plot(selled)
plt.xlabel("Количество шагов")
plt.ylabel("Количество проданных квартир")
plt.savefig('main/pictures/sold_e_greedy.png')
plt.show()