import numpy as np
import pandas as pd
import category_encoders as ce
import re
import math
from sklearn.utils import shuffle
import datetime

def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 10
    display.max_rows = 1000
    display.max_colwidth = 40
    display.width = 1000
    # display.precision = 2  # set as needed


def to_1D(series):
 return pd.Series([x for _list in series for x in _list])

def ret_value_counts(data, str):
    return to_1D(data[str]).value_counts()


def boolean_df(item_lists, unique_items):
# Create empty dict
    bool_dict = {}
    
    # Loop through all the tags
    for i, item in enumerate(unique_items):
        
        # Apply boolean mask
        bool_dict[item] = item_lists.apply(lambda x: item in x)
            
    # Return the results as a dataframe
    return pd.DataFrame(bool_dict)


def mean_target_encoding(df, target, column):
    mean_enc = df.groupby(column)[target].mean()
    df[column+'_m_enc'] = df[column].map(mean_enc)
    return (df)



def parse_subways_target_enc(data, target):
    data["address_subways"] = data["address_subways"].fillna("[]").apply(lambda x: eval(x))
    
    num_subway_st = list()
    num_subway_lines = list()
    subway_lines = list()
    subway_st = list()
    min_distance_subway = list()
    min_time_subway = list()

    num_subway_st = list(map(lambda x: (len(x)), data['address_subways']))
    data['num_subway_st'] = num_subway_st

    for elem in data['address_subways']:
        min_d = 10000000000000
        min_t = 10000000000000
        subway_st_temp = list()
        subway_lines_temp = list()
        for x in elem:
            st, dist, time, line = x
            subway_st_temp.append(st)
            subway_lines_temp.append(line)
            min_d = min(min_d, dist)
            min_t = min(min_t, time)
        min_distance_subway.append(min_d)
        min_time_subway.append(min_t)
        subway_lines_temp = np.unique(subway_lines_temp)
        subway_lines.append(subway_lines_temp)
        subway_st.append(subway_st_temp)

    data['min_distance_subway'] = min_distance_subway
    data['min_time_subway'] = min_time_subway
    data['subway_lines'] = subway_lines
    data['subway_st'] = subway_st
    del data['address_subways']

    num_subway_lines = list(map(lambda x: (len(x)), data['subway_lines']))
    data['num_subway_lines'] = num_subway_lines

    #now let's deal with lists

    s_lines = ret_value_counts(data, 'subway_lines')
    #s_lines = s_lines.add_suffix("_метро")
    df2 = boolean_df(data['subway_lines'], s_lines.keys())
    df2 = pd.concat([data[target], df2], axis=1)
    for line in s_lines.keys():
        mean_enc = df2.groupby(line)[target].mean()
        df2[line] = df2[line].map(mean_enc)
    del df2[target]
    data['subway_lines'] = np.log(df2.sum(axis=1))

    s_lines = ret_value_counts(data, 'subway_st')
    #s_lines = s_lines.add_suffix("_st")
    df2 = boolean_df(data['subway_st'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['subway_st']


    # s_lines = ret_value_counts(data, 'subway_st')
    # #s_lines = s_lines.add_suffix("_st")
    # df2 = boolean_df(data['subway_st'], s_lines.keys())
    # df2 = pd.concat([data[target], df2], axis=1)
    # for st in s_lines.keys():
    #     mean_enc = df2.groupby(st)[target].mean()
    #     df2[line] = df2[st].map(mean_enc)
    # del df2[target]
    # data['subway_st'] = np.log(df2.sum(axis=1))


    return data


def create_area_category(data):
    status = list()
    for area in data["obj_area"]:
        if area > 600:
            status.append("1")
        elif area > 400:
            status.append("2")
        elif area >200:
            status.append("3")
        elif area >150:
            status.append("4")
        elif area>100:
            status.append("5")
        elif area > 75:
            status.append("6")
        elif area >50:
            status.append("7")
        elif area >25:
            status.append("8")
        else:
            status.append("9")
    data["obj_area_cat"] = status
    return data

def target_area_category(data, target):
    s_lines = ret_value_counts(data, 'obj_area_cat')
    df2 = boolean_df(data['obj_area_cat'], s_lines.keys())
    df2 = pd.concat([data[target], df2], axis=1)
    for line in s_lines.keys():
        mean_enc = df2.groupby(line)[target].mean()
        df2[line] = df2[line].map(mean_enc)
    del df2[target]
    data['obj_area_cat_t'] = np.log(df2.sum(axis=1))
    del data['obj_area_cat']
    return data

####################################################################################################################################################################


def parse_subways_json(data):
    name = "Метро"
    data[name] = data[name].fillna("[]")
    num_subway_st = list()
    subway_st = list()
    min_time_subway = list()
    nearest_subway_st = list()

    num_subway_st = list(map(lambda x: (len(x)), data[name]))
    data['num_subway_st'] = num_subway_st

    for elem in data[name]:
        min_t = 10000000000000
        near = ""
        subway_st_temp = list()
        for x in elem:
            n = list(x.keys())[0]
            t = list(x.values())[0]
            times = re.split("-|–| ", t)
            t = list()
            for word in times:
                if word.isdigit():
                    t.append(int(word))
            if len(t) == 2:
                time = (t[0] + t[1])/2
            else:
                time = t[0]
            subway_st_temp.append(n)
            if time < min_t:
                min_t = time
                near = n
        min_time_subway.append(min_t)
        subway_st.append(subway_st_temp)
        nearest_subway_st.append(near)

    data['min_time_subway'] = min_time_subway
    data['subway_st'] = subway_st
    data['nearest_subway_st'] = nearest_subway_st
    del data[name]

    s_lines = ret_value_counts(data, 'subway_st')
    #s_lines = s_lines.add_suffix("_st")
    df2 = boolean_df(data['subway_st'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['subway_st']
    f = open("/home/maryna/dynamic-pricing/models/new_json/subway.txt")
    subway_list = f.readlines()
    for st in subway_list:
        st2 = st.strip()
        if st2 not in data:
            data[st2] = False
    return data

def target_enc_subways(data, target):
    name = "Метро"
    data[name] = data[name].fillna("[]")
    num_subway_st = list()
    subway_st = list()
    min_time_subway = list()
    nearest_subway_st = list()

    num_subway_st = list(map(lambda x: (len(x)), data[name]))
    data['num_subway_st'] = num_subway_st

    for elem in data[name]:
        min_t = 10000000000000
        near = ""
        subway_st_temp = list()
        for x in elem:
            n = list(x.keys())[0]
            t = list(x.values())[0]
            times = re.split("-|–| ", t)
            t = list()
            for word in times:
                if word.isdigit():
                    t.append(int(word))
            if len(t) == 2:
                time = (t[0] + t[1])/2
            else:
                time = t[0]
            subway_st_temp.append(n)
            if time < min_t:
                min_t = time
                near = n
        min_time_subway.append(min_t)
        subway_st.append(subway_st_temp)
        nearest_subway_st.append(near)

    data['min_time_subway'] = min_time_subway
    data['subway_st'] = subway_st
    data['nearest_subway_st'] = nearest_subway_st
    del data[name]
    s_lines = ret_value_counts(data, 'subway_st')
    df2 = boolean_df(data['subway_st'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    f = open("/home/maryna/dynamic-pricing/models/new_json/subway.txt")
    subway_list = f.readlines()
    for st in subway_list:
        st2 = st.strip()
        if st2 not in data:
            data[st2] = False
    
    subway_price = dict()

    df2 = pd.concat([data[target], df2], axis=1)
    for line in s_lines.keys():
        mean_enc = df2.groupby(line)[target].mean()
        subway_price[line] = mean_enc
        df2[line] = df2[line].map(mean_enc)
    print(df2)
    del df2[target]
    data['subway_st'] = np.log(df2.sum(axis=1))
    # print(subway_price)
    return data

def parse_yard_json(data):
    name = "Двор"
    yards = []

    for elem in data[name]:
        if not isinstance(elem, str):
            yards.append([])
        else:
            e = re.split(",", elem)
            e = list(map(lambda x: x.strip(), e))
            yards.append(e)
    data["yards"] = yards
    del data[name]


    #now let's deal with lists

    s_lines = ret_value_counts(data, 'yards')
    #s_lines = s_lines.add_suffix("_st")
    df2 = boolean_df(data['yards'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['yards']

    return data

def convert_to_unixtime(date_string):
    date_object = datetime.datetime.strptime(date_string, '%d.%m.%Y')
    return date_object.timestamp()

def parse_keyrate(data, keyrate):
    data["keyrate"] = [0] * len(data)
    i = 0
    for i in range(len(data)):
        t = data['Дата выхода (unixtime)'][i]
        t_plus_1_day = datetime.datetime.fromtimestamp(t) + datetime.timedelta(days=1)
        t_plus_1_day = int(t_plus_1_day.timestamp())
        t_plus_2_day = datetime.datetime.fromtimestamp(t) + datetime.timedelta(days=2)
        t_plus_2_day = int(t_plus_2_day.timestamp())
        t_minus_1_day = datetime.datetime.fromtimestamp(t) - datetime.timedelta(days=1)
        t_minus_1_day = int(t_minus_1_day.timestamp())
        t_minus_2_day = datetime.datetime.fromtimestamp(t) - datetime.timedelta(days=2)
        t_minus_2_day = int(t_minus_2_day.timestamp())
        if t in keyrate.index:
            data['keyrate'][i] = keyrate["key"][t]
            # print(1)
        elif t_plus_1_day in keyrate.index:
            data['keyrate'][i] = keyrate["key"][t_plus_1_day]
            # print(2)
        elif t_plus_2_day in keyrate.index:
            data['keyrate'][i] = keyrate["key"][t_plus_2_day]
            # print(3)
        elif t_minus_1_day in keyrate.index:
            data['keyrate'][i] = keyrate["key"][t_minus_1_day]
            # print(4)
        elif t_minus_2_day in keyrate.index:
            data['keyrate'][i] = keyrate["key"][t_minus_2_day]
            # print(5)
        else:
            data['keyrate'][i] = 0
            # print(0)
    return data

def parse_currency(data, currency_str, currency):
    data[currency_str] = [0] * len(data)
    i = 0
    for i in range(len(data)):
        t = data['Дата выхода (unixtime)'][i]
        t_plus_1_day = datetime.datetime.fromtimestamp(t) + datetime.timedelta(days=1)
        t_plus_1_day = int(t_plus_1_day.timestamp())
        t_plus_2_day = datetime.datetime.fromtimestamp(t) + datetime.timedelta(days=2)
        t_plus_2_day = int(t_plus_2_day.timestamp())
        t_minus_1_day = datetime.datetime.fromtimestamp(t) - datetime.timedelta(days=1)
        t_minus_1_day = int(t_minus_1_day.timestamp())
        t_minus_2_day = datetime.datetime.fromtimestamp(t) - datetime.timedelta(days=2)
        t_minus_2_day = int(t_minus_2_day.timestamp())
        if t in currency.index:
            data[currency_str][i] = currency["num"][t]
            # print(1)
        elif t_plus_1_day in currency.index:
            data[currency_str][i] = currency["num"][t_plus_1_day]
            # print(2)
        elif t_plus_2_day in currency.index:
            data[currency_str][i] = currency["num"][t_plus_2_day]
            # print(3)
        elif t_minus_1_day in currency.index:
            data[currency_str][i] = currency["num"][t_minus_1_day]
            # print(4)
        elif t_minus_2_day in currency.index:
            data[currency_str][i] = currency["num"][t_minus_2_day]
            # print(5)
        else:
            print(datetime.datetime.fromtimestamp(t))
            data[currency_str][i] = 0
            # print(0)
    return data

def parse_date(data):
    for i in range(len(data)):
        t = datetime.datetime.fromtimestamp(data['Дата выхода (unixtime)'][i])
        t = t.replace(hour=0, minute=0, second=0)
        data['Дата выхода (unixtime)'][i] = int(t.timestamp())
    return data

def prepare_all(data):
    data = shuffle(data)
    data = parse_yard_json(data)
    set_to_left = set(['Разница в днях', "Id", "Просмотры", 'Дата выхода (unixtime)', 'Балкон или лоджия', 'Окна', 'Отделка', 'Название новостройки', 'Официальный застройщик', 'Тип дома', 'Этажей в доме', 'Цена', 'Метро', 'Общая площадь (fixed)', 'Площадь кухни (fixed)', 'Жилая площадь (fixed)', 'Этаж (fixed)', 'Высота потолков (fixed)', 'Срок сдачи (fixed)', 'Санузел', 'Парковка', 'Пассажирский лифт (fixed)', 'Грузовой лифт (fixed)', 'Тип комнат', 'Год постройки', 'детская площадка', 'закрытая территория', 'спортивная площадка'])
    for field in data:
        if field not in set_to_left:
            data.drop([field], axis=1, inplace=True)
    data = parse_date(data)
    keyrate = pd.read_csv('/home/maryna/dynamic-pricing/models/data/last/keyratedata.csv')
    keyrate['date'] =keyrate['date'].apply(convert_to_unixtime)
    keyrate = keyrate.set_index('date')
    data = parse_keyrate(data, keyrate)
    currency = pd.read_csv("/home/maryna/dynamic-pricing/models/data/last/currency_data.csv", usecols=['date', 'currency', 'num'])
    currency['date'] = currency['date'].apply(convert_to_unixtime)
    currency = currency.set_index('date')
    value_to_split = 'USD'
    USD = currency.loc[currency['currency'] == value_to_split]
    USD.drop(['currency'], axis=1, inplace=True)
    EUR = currency.loc[currency['currency'] != value_to_split]
    EUR.drop(['currency'], axis=1, inplace=True)
    data = parse_currency(data, 'USD', USD)
    data = parse_currency(data, 'EUR', EUR)

    list_for_drop = ['Дата парсинга', 'Дата выхода (unixtime)', 'Url', 'Количество комнат', 'Общая площадь', 'Площадь кухни', 'Жилая площадь', 'Этаж', 'Высота потолков', 'Срок сдачи', 'Дата выхода', 'Вид сделки', "Способ продажи", "Всего этажей", "Корпус, строение", "Тип участия"]
    for name in list_for_drop:
        if name in data:
            data.drop([name], axis=1, inplace=True)
    

    list_for_factorize = ["Отделка", "Санузел", "Парковка", "Балкон или лоджия", "Тип комнат", "Название новостройки", "Официальный застройщик", "Тип дома", "Срок сдачи (fixed)", "Окна", 'nearest_subway_st']
    for name in list_for_factorize:
        if name in data:
            data[name] = pd.factorize(data[name])[0]
    
    data['Разница в днях'] = data['Разница в днях'].replace(0, 0.5)
    data["view_av"] = data["Просмотры"] / data["Разница в днях"]
    data.drop(data[data["view_av"] > 500].index, inplace=True)
    data.drop(data[data["Общая площадь (fixed)"] > 5000].index, inplace=True)
    del data["Просмотры"]
    del data["Id"]
    del data['Разница в днях']
    data = parse_subways_json(data)
    
    return data

data = pd.read_csv('/home/maryna/dynamic-pricing/all_ready_data.csv')
print(np.mean(data['Цена']))


# # print(data.dtypes)

# # print(data['Дата парсинга'])
# print(data.iloc[24306])
# data = prepare_all(data)
# data.to_csv('all_ready_data.csv', index=False)
# data = prepare_all(data)
# print(len(data))
# print(data.dtypes)

# print(data['view_av'].value_counts())

# # duplicateRows = data[data.duplicated()]
# # print("DUPLICATEs", duplicateRows)

# # print("Unique", data.nunique())

# print(data["Тёплый пол"])
# import matplotlib.pyplot as plt
# plt.hist(list(np.sort(data["view_av"])), bins=200)
# plt.xlabel('Среднесуточное значение (просмотры)')
# plt.ylabel('Частота')
# plt.show()

