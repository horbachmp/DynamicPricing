data_path = "main/data/all_ready_data.csv"
start_index = 20000 #начала среза данных
m = 1000  #- количество ручек
n = 10000 # количество квартир
start_dispersion = 500000
selling_treshold = 30 # - количество просмотров, на которое прихоодится одна покупка
T = 100 # кол-во шагов



import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import helpers.demand_class as demand
import helpers.start_price_class as start_price
import time

start_time = time.time()
results = list()
selled = list()


data = pd.read_csv(data_path)[start_index:start_index+n]
data_y  = data['Цена'].values

f = open("main/logs/real_price_gauss.txt", "w")
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


def evaluate(p, data_t):
    d2 = data_t.copy()
    d2["Цена"] = p
    d2.drop(['view_av'], axis=1, inplace=True)
    preds = np.clip(demand_model.predict(d2), 0, 5000)
    # print(preds)
    R = np.round(preds/selling_treshold) # продана/не продано
    return R, preds

def get_real_profit(p, data_t):
    d2 = data_t.copy()
    d2["Цена"] = p
    d2.drop(['view_av'], axis=1, inplace=True)
    preds = np.clip(demand_model.predict(d2), 0, 5000)
    f = np.round(preds/selling_treshold) 
    R = f * p
    return np.sum(R)



class BanditArm():
    def __init__(self, price, m) -> None:
        self.a = m
        self.b = 1
        self.n = 0
        self.price = price
        self.mu = m

    def Sample(self):
        return np.random.gamma(self.a, self.b)
    

    def Play(self, x): # x - спрос в просмотрах
        self.n += 1 
        self.a += 1/2
        self.b += ((x - self.mu) ** 2)/2

class Bandit():
    def __init__(self, m, prices, mean_v) -> None:
        self.len_arms = m
        self.arms = [BanditArm(prices[i], mean_v[i]) for i in range(self.len_arms)]
        self.curr_play = -1
    
    def Step(self):
        samples = np.array([arm.Sample() for arm in self.arms])
        # print("samples:", samples)
        self.curr_play = np.argmax(samples)
        return self.arms[self.curr_play].price
    
    def Update(self, view):
        self.arms[self.curr_play].Play(view)



# создаем ручки (цены) для всех квартир
df = data.copy()
df.drop(['Цена'], axis=1,inplace=True)
df.drop(['view_av'], axis=1,inplace=True)
# print(df.dtypes)
start_prices = start_price_model.predict(df)

R, start_views = evaluate(start_prices, data)

print("Start generating")
batdits = list()
for i in range(n):
    prices_one = np.random.normal(start_prices[i], start_dispersion, m)
    batdits.append(Bandit(m, prices_one, [start_views[i]] * m))

print("Generated")

curr_prices = start_prices
start_cycle_time = time.time()
for t in range(T):
    print(t, "/", T)
    curr_prices = [bandit.Step() for bandit in batdits]
    R, views = evaluate(curr_prices, data)
    for i in range(n):
        batdits[i].Update(views[i])
    results.append(get_real_profit(curr_prices, data))
    print("Продано:", np.sum(R))
    selled.append(np.sum(R))


results.append(get_real_profit(curr_prices, data))


end_time = time.time()

print("Среднее время исполнения шага:", (end_time - start_cycle_time)/T)
print("Время выполнения", end_time - start_time)
print("Прибыль", results[-1])

f2 = open("main/logs/final_price_gauss.txt", "w")
for p in curr_prices:
    print(p, file=f2)

print("MAE = ", mean_absolute_error(data_y, curr_prices))
print("MSE = ", mean_squared_error(data_y, curr_prices))
print("MAPE = ", 100 * mean_absolute_percentage_error(data_y, curr_prices))
print("ME = ", sum(curr_prices-data_y)/len(data_y))


plt.plot(results)
plt.xlabel("Количество шагов")
plt.ylabel("Прибыль")
plt.savefig('main/pictures/results_gauss.png')
plt.show()
plt.plot(selled)
plt.xlabel("Количество шагов")
plt.ylabel("Количество проданных квартир")
plt.savefig('main/pictures/sold_gauss.png')
plt.show()

