import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import helpers.demand_class as demand

demand_model = demand.DemandModel("/home/maryna/dynamic-pricing/demand.model")

data = pd.read_csv('/home/maryna/dynamic-pricing/models/data/cian_new_flats.csv', delimiter=";")
data.drop(data[data["totalArea"] > 5000].index, inplace=True)
data.drop(data[np.isnan(data["viewCountTotal"])].index, inplace=True)

data_y  = data['viewCountTotal'].values
data.drop(['viewCountTotal'], axis=1, inplace=True)
train_df, test_df, train_y, test_y = train_test_split(data, data_y, test_size=0.1, random_state=42)
test_ids = test_df['offerId']

train_df.drop(['offerId'], axis=1, inplace=True)
test_df.drop(['offerId'], axis=1, inplace=True)

df = pd.concat([train_df, test_df])

df_num = df.select_dtypes(exclude=['object'])
df_obj = df.select_dtypes(include=['object']).copy()
for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_num, df_obj], axis=1)
pos = train_df.shape[0]
train_df = df_values[:pos]
test_df  = df_values[pos:]
del df, df_num, df_obj, df_values


preds = demand_model.predict(test_df)
exp_preds = np.exp(preds)
exp_test_y = np.exp(test_y)
print("MAE = ", mean_absolute_error(exp_preds,exp_test_y))
print("MSE = ", mean_squared_error(exp_preds,exp_test_y))
print("MAPE = ", 100 * mean_absolute_percentage_error(exp_preds,exp_test_y))
print("ME = ", sum(exp_test_y - exp_preds)/len(exp_test_y))
