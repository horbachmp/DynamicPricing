data_path = "main/data/all_ready_data.csv"
n = 20000 # размер обучающей выборки

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

RS = 20170501
np.random.seed(RS)

ROUNDS = 1000
params = {
	'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.08,
    'verbose': 0,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': RS,
    'feature_fraction': 0.7,
    'feature_fraction_seed': RS,
    'max_bin': 100,
    'max_depth': 6,
    'num_rounds': ROUNDS
}

print("Started")
data = pd.read_csv(data_path)[:n]

data_y  = data['Цена'].values
data.drop(['Цена'], axis=1, inplace=True)
data.drop(['view_av'], axis=1, inplace=True)



train_df, test_df, train_y, test_y = train_test_split(data, data_y, test_size=0.05, random_state=42)
print(train_df.shape, test_df.shape, train_y.shape, test_y.shape)
print("Data: X_train: {}, X_test: {}".format(train_df.shape, test_df.shape))

df = pd.concat([train_df, test_df])


df_num = df.select_dtypes(exclude=['object'])
df_obj = df.select_dtypes(include=['object']).copy()
for c in df_obj:
    print("Warning: field", c, "needs factorization")
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_num, df_obj], axis=1)


pos = train_df.shape[0]
train_df = df_values[:pos]
test_df  = df_values[pos:]
del df, df_num, df_obj, df_values

print("Training on: {}".format(train_df.shape, train_y.shape))


train_lgb = lgb.Dataset(train_df, train_y)
model = lgb.train(params, train_lgb, num_boost_round=ROUNDS)
preds = model.predict(test_df)
	


print("Features importance...")
gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(25))



plt.figure()
ft[['feature','gain']].head(25).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
plt.gcf().savefig('main/pictures/features_importance_start_price.png')

print("Done.")

print("MAE = ", mean_absolute_error(preds, test_y))
print("MSE = ", mean_squared_error(preds,test_y))
print("MAPE = ", 100 * mean_absolute_percentage_error(preds,test_y))
print("ME = ", sum(test_y - preds)/len(test_y))

print("Saving model ...")
model.save_model("main/models/weights/start_price.model")
print("Saved")