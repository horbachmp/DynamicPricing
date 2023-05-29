import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import helpers.demand_class as demand
from scipy.stats import bernoulli
import time

start_time = time.time()

m = 1000  #- количество ручек
n = 10000 # количество квартир
start_dispersion = 1000000 # - для генерации ручек
selling_treshold = 30 # - количество просмотров, на которое прихоодится одна покупка
T = 1000 # кол-во шагов
epsilon = 0.3
results = list()
selled = list()



data = pd.read_csv('/home/maryna/dynamic-pricing/all_ready_data.csv')[20000:20000+n] # берем срез, чтобы данные из обучения не попали в тест
data_y  = data['Цена'].values

f = open("real_price.txt", "w")
for p in list(data_y):
    print(p, file=f)


df_num = data.select_dtypes(exclude=['object'])
df_obj = data.select_dtypes(include=['object']).copy()
for c in df_obj:
    print("Warning: some not factorized field:", c)
    df_obj[c] = pd.factorize(df_obj[c])[0]
data = pd.concat([df_num, df_obj], axis=1)

demand_model = demand.DemandModel("demand.model")




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

start_prices = np.ones(n) * 15323486
R, s = get_real_profit(start_prices, data)
results.append(sum(R))
selled.append(s)


all_prices = list()
for i in range(n):
    prices_one = np.random.normal(15323486, start_dispersion, m)
    all_prices.append(prices_one)


mean_rewards = np.zeros((n, m))
num_pulling = np.zeros((n, m))
curr_prices = start_prices
print(curr_prices)
start_cycle_time = time.time()
for t in range(T):
    print(t, "/", T)
    print(s)
    # print(curr_prices)
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

f2 = open("final_price.txt", "w")
for p in curr_prices:
    print(p, file=f2)


print("MAE = ", mean_absolute_error(data_y, curr_prices))
print("MSE = ", mean_squared_error(data_y, curr_prices))
print("MAPE = ", 100 * mean_absolute_percentage_error(data_y, curr_prices))
print("ME = ", sum(curr_prices-data_y)/len(data_y))

print("Среднее время исполнения шага:", (end_time - start_cycle_time)/T)
print("Время выполнения", end_time - start_time)
print("прибыль", results[-1])
plt.plot(results)
plt.xlabel("Количество шагов")
plt.ylabel("Прибыль")
plt.savefig('results_e_greedy_no_start_price.png')
plt.show()
plt.plot(selled)
plt.xlabel("Количество шагов")
plt.ylabel("Количество проданных квартир")
plt.savefig('selled_e_greedy_no_start_price.png')
plt.show()