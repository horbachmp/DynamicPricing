import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import helpers.demand_class as demand
import helpers.start_price_class as start_price
import time



m = 1000  #- количество ручек
n = 10000 # количество квартир
start_dispersion = 500000
selling_treshold = 30 # - количество просмотров, на которое прихоодится одна покупка
T = 1000 # кол-во шагов
results = list()
selled = list()




data = pd.read_csv('/home/maryna/dynamic-pricing/all_ready_data.csv')[20000:20000+n] # берем срез, чтобы данные из обучения не попали в тест
data_y  = data['Цена'].values

df_num = data.select_dtypes(exclude=['object'])
df_obj = data.select_dtypes(include=['object']).copy()
for c in df_obj:
    print("Warning: some not factorized field:", c)
    df_obj[c] = pd.factorize(df_obj[c])[0]
data = pd.concat([df_num, df_obj], axis=1)



start_price_model = start_price.StartPriceModel("start_price_new.model")
demand_model = demand.DemandModel("demand.model")


def evaluate(p, data_t):
    d2 = data_t.copy()
    d2["Цена"] = p
    d2.drop(['view_av'], axis=1, inplace=True)
    preds = np.clip(demand_model.predict(d2), 0, 5000)
    R = np.array(preds>selling_treshold) # продана/не продано
    return R

def get_real_profit(p, data_t):
    d2 = data_t.copy()
    d2["Цена"] = p
    d2.drop(['view_av'], axis=1, inplace=True)
    preds = np.clip(demand_model.predict(d2), 0, 5000)
    f = np.round(preds/selling_treshold) 
    R = f * p
    return np.sum(R), sum(f)




# создаем ручки (цены) для всех квартир
df = data.copy()
df.drop(['Цена'], axis=1,inplace=True)
df.drop(['view_av'], axis=1,inplace=True)
start_prices = start_price_model.predict(df)

revenue, s = get_real_profit(start_prices, data)
print(s)


all_prices = list()
for i in range(n):
    prices_one = np.random.normal(start_prices[i], start_dispersion, m)
    # while not (np.all(prices_one > start_prices[i] - 2 * start_dispersion) and np.all(prices_one < start_prices[i] + 2 * start_dispersion)):
    #     prices_one = np.random.normal(start_prices[i], start_dispersion, m)
    all_prices.append(prices_one)

print("Generated")

alphas = np.ones((n,m))
betas = np.ones((n,m))
curr_prices = start_prices
print(curr_prices)

for t in range(T):
    print(t, "/", T)
    revenue, s = get_real_profit(curr_prices, data)
    results.append(revenue)
    selled.append(s)
    print(s)
    # print(curr_prices)
    pull_nums = list()
    for i in range(n): # для каждой квартиры
        samples = list(map(lambda x: beta.rvs(x[0], x[1]), zip(alphas[i], betas[i])))
        pull_num = np.argmax(np.array(samples))
        curr_prices[i] = all_prices[i][pull_num]
        pull_nums.append(pull_num)
    R = evaluate(curr_prices, data)
    for i in range(n):
        alphas[i][pull_nums[i]] += R[i]
        betas[i][pull_nums[i]] += (1-R[i])



plt.plot(results)
plt.xlabel("Количество шагов")
plt.ylabel("Прибыль")
plt.savefig('results_bernulli.png')
plt.show()
plt.plot(selled)
plt.xlabel("Количество шагов")
plt.ylabel("Количество проданных квартир")
plt.savefig('selled_bernulli.png')
plt.show()
