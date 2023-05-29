import numpy as np
import pandas as pd
import category_encoders as ce


def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 10
    display.max_rows = 1000
    display.max_colwidth = 40
    display.width = 1000
    # display.precision = 2  # set as needed

# set_pandas_display_options()

# def parse_lists(data):
#     data["house_infrastructure"] = data["house_infrastructure"].fillna("[]").apply(lambda x: eval(x))
#     data["house_yard"] = data["house_yard"].fillna("[]").apply(lambda x: eval(x))
#     data["obj_window_view"] = data["obj_window_view"].fillna("[]").apply(lambda x: eval(x))
#     return data


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


#process subways
def parse_subways(data):
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
    data = pd.concat([data, df2], axis=1)
    del data['subway_lines']

    s_lines = ret_value_counts(data, 'subway_st')
    #s_lines = s_lines.add_suffix("_st")
    df2 = boolean_df(data['subway_st'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['subway_st']

    return data



#process infrastructure
def parse_infrastructure(data):
    data["house_infrastructure"] = data["house_infrastructure"].fillna("[]").apply(lambda x: eval(x))

    #now let's deal with lists

    for i in range(len(data["house_infrastructure"])):
        for j in range(len(data["house_infrastructure"][i])):
            if data["house_infrastructure"][i][j] == "Детскийсад":
                data["house_infrastructure"][i][j] = "Детский сад"
            elif data["house_infrastructure"][i][j] == "Торговыйцентр":
                data["house_infrastructure"][i][j] = "Торговый центр"


    s_lines = ret_value_counts(data, 'house_infrastructure')
    s_lines = s_lines.add_suffix(" инфра")
    df2 = boolean_df(data['house_infrastructure'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['house_infrastructure']

    return data

def if_num_else_nan(arr, ind):
    if len(arr)>ind:
        return arr[ind]
    return "nan"


#process address
def parse_address(data):

    data["address_name"] = list(map(lambda x: x.split(','), data["address_name"]))

    data["country"] = list(map(lambda x: if_num_else_nan(x, 0), data["address_name"]))
    data["city"] = list(map(lambda x: if_num_else_nan(x, 1), data["address_name"]))
    data["street"] = list(map(lambda x: if_num_else_nan(x, 2), data["address_name"]))
    data["house"] = list(map(lambda x: if_num_else_nan(x, 3), data["address_name"]))

    del data['address_name']

    #now let's deal with lists


    # s_lines = ret_value_counts(data, 'house_infrastructure')
    # df2 = boolean_df(data['house_infrastructure'], s_lines.keys())
    # data = pd.concat([data, df2], axis=1)
    # del data['house_infrastructure']

    return data


#process house_yard
def parse_house_yard(data):
    data["house_yard"] = data["house_yard"].fillna("[]").apply(lambda x: eval(x))

    for i in range(len(data["house_yard"])):
        for j in range(len(data["house_yard"][i])):
            if data["house_yard"][i][j] == "Детскаяплощадка":
                data["house_yard"][i][j] = "Детская площадка"
            elif data["house_yard"][i][j] == "Спортивнаяплощадка":
                data["house_yard"][i][j] = "Спортивная площадка"

    #now let's deal with lists


    s_lines = ret_value_counts(data, 'house_yard')
    s_lines = s_lines.add_suffix(" двор")
    df2 = boolean_df(data['house_yard'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['house_yard']

    return data


#process views
def parse_views(data):
    data["obj_window_view"] = data["obj_window_view"].fillna("[]").apply(lambda x: eval(x))

    # print(max(map(lambda x: len(x), data["obj_window_view"])))

    #now let's deal with lists


    s_lines = ret_value_counts(data, 'obj_window_view')
    s_lines = s_lines.add_suffix(" вид")
    df2 = boolean_df(data['obj_window_view'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['obj_window_view']

    return data




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
    print(s_lines)
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

def parse_date(data):
    #TODO
    return data

def parse_highways(data):
    data["highways"] = data["highways"].fillna("[]").apply(lambda x: eval(x))
    highways = list()
    highway_ids = list()
    min_distance_highway = list()

    num_highways = list(map(lambda x: (len(x)), data['highways']))
    data['num_highways'] = num_highways

    for elem in data['highways']:
        min_d = 10000
        highways_temp = list()
        ids_temp = list()
        for x in elem:
            name = x['name']
            dist = x['distance']
            url = x['url']
            id = x['id']
            highways_temp.append(name)
            ids_temp.append(id)
            min_d = min(min_d, dist)
        min_distance_highway.append(min_d)
        highways_temp = np.unique(highways_temp)
        highways.append(highways_temp)
        ids_temp = np.unique(ids_temp)
        highway_ids.append(ids_temp)

    data['min_distance_highway'] = min_distance_highway
    # data['highways'] = highways
    data['highway_ids'] = highway_ids
    del data['highways']

    #now let's deal with lists
    s_lines = ret_value_counts(data, 'highway_ids')
    s_lines = s_lines.add_suffix("_highway")
    df2 = boolean_df(data['highway_ids'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['highway_ids']

    return data

def parse_coordinates(data):
    data["coordinates"] = data["coordinates"].fillna("{}").apply(lambda x: eval(x))
    data["coordinates_lat"] = list(map(lambda x: x['lat'], data["coordinates"]))
    data["coordinates_lng"] = list(map(lambda x: x['lng'], data["coordinates"]))
    del data["coordinates"]
    return data

def parse_railways(data):
    data["railways"] = data["railways"].fillna("[]").apply(lambda x: eval(x))
    railways = list()
    railways_ids = list()
    min_distance_railways = list()

    num = list(map(lambda x: (len(x)), data['railways']))
    data['num_railways'] = num

    for elem in data['railways']:
        min_d = 10000
        temp = list()
        ids_temp = list()
        for x in elem:
            name = x['name']
            dist = float(x['distance'])
            id = x['id']
            temp.append(name)
            ids_temp.append(id)
            min_d = min(min_d, dist)
        min_distance_railways.append(min_d)
        temp = np.unique(temp)
        railways.append(temp)
        ids_temp = np.unique(ids_temp)
        railways_ids.append(ids_temp)

    data['min_distance_railway'] = min_distance_railways
    data['railways_ids'] = railways_ids
    del data['railways']

    #now let's deal with lists
    s_lines = ret_value_counts(data, 'railways_ids')
    s_lines = s_lines.add_suffix("_railway")
    df2 = boolean_df(data['railways_ids'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['railways_ids']

    return data


def parse_undergrounds(data):
    data["undergrounds"] = data["undergrounds"].fillna("[]").apply(lambda x: eval(x))
    undergrounds = list()
    min_time_undergrounds = list()
    lines = list()
    nearest_stations = list()
    nearest_lines = list()

    num = list(map(lambda x: (len(x)), data['undergrounds']))
    data['num_undergrounds'] = num

    color_lines = {'CF0000':1, '00701A' : 2, '03238B':3, '009BD5':4, '800000' : 5, 'FF7F00' : 6, '94007C': 7, 'FFDF00':8, 'A2A5B5':9, '8AD02A':10, '7ACDCE':11, 'FA6F9E':15}
 

    for elem in data['undergrounds']:
        min_t = 10000
        temp = list()
        lines_temp = list()
        for x in elem:
            name = x['name']
            time = float(x['travelTime'])
            color = x['lineColor']
            temp.append(name)
            number = color_lines[color]
            if time < min_t:
                nearest_line = number
                nearest_station = name
                nearest_station
                min_t = time
            lines_temp.append(number)
        min_time_undergrounds.append(min_t)
        temp = np.unique(temp)
        lines_temp = np.unique(lines_temp)
        undergrounds.append(temp)
        lines.append(lines_temp)
        nearest_lines.append(nearest_line)
        nearest_stations.append(nearest_station)

    data['min_time_undergrounds'] = min_time_undergrounds
    data['undergrounds'] = undergrounds
    data['lines'] = lines
    data["nearest_station"] = nearest_stations
    data['nearest_line'] = nearest_lines

    #now let's deal with lists
    s_lines = ret_value_counts(data, 'undergrounds')
    s_lines = s_lines.add_suffix("_underground")
    df2 = boolean_df(data['undergrounds'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['undergrounds']

    s_lines = ret_value_counts(data, 'lines')
    s_lines = s_lines.add_suffix("_line")
    df2 = boolean_df(data['lines'], s_lines.keys())
    data = pd.concat([data, df2], axis=1)
    del data['lines']

    return data


def parse_deadlineQuarter(data):
    i = 0
    deadline = list()
    for x in data['deadlineQuarter']:
        if x == "first":
            deadline.append(1)
        elif x == "second":
            deadline.append(2)
        elif x == "third":
            deadline.append(3)
        elif x == "fourth":
            deadline.append(4)
        else:
            deadline.append(None)
    data['deadlineQuarter'] = deadline
    return data

def parse_adress(data):
    type2 = data["address"].fillna("[]").apply(lambda x: eval(x))
    city = list()
    district = list()
    street = list()
    area = list()

    for elem in type2:
        elem_dict = {}
        for item in elem:
            elem_dict[item['type']] = item['fullName']
        if "street" in elem_dict:
            street.append(elem_dict["street"])
        else:
            street.append(None)
        if "okrug" in elem_dict:
            district.append(elem_dict["okrug"])
        else:
            district.append(None)
        if "raion" in elem_dict:
            area.append(elem_dict["raion"])
        else:
            area.append(None)
        elem2 = [[d['fullName'], d['type']] for d in elem]
        if elem2[0][1] == "location":
            city.append(elem2[0][0])
        else:
            city.append(None)
            print("erroe", elem2[0])

    data['city'] = city
    data['area'] = area
    data['district'] = district
    data["street"] = street
    del data["address"]
    return data




def prepare_all(data):
    #fix outlier
    data.drop(data[data["totalArea"] > 5000].index, inplace=True)
    # data.drop(data[np.isnan(data["viewCountTotal"])].index, inplace=True)

    data = parse_adress(data)
    data = parse_deadlineQuarter(data)
    data = parse_undergrounds(data)
    data = parse_railways(data)
    data = parse_coordinates(data)
    data = parse_highways(data)

    return data

# data = pd.read_csv('/home/maryna/dynamic-pricing/models/data/cian_new_flats.csv', delimiter=";")

# data = parse_highways(data)
# data = parse_coordinates(data)
# data = parse_railways(data)
# data = parse_undergrounds(data)
# data = parse_deadlineQuarter(data)
# data = parse_adress(data)

# for x in range(0, 51, 10):
#     print(data.dtypes[x:x+10])
# print(data.dtypes[55:])

# for i in data["newObjectInfrastructureTypes"]:
#     if i != data["newObjectInfrastructureTypes"][0]:                                                               
#         print(i)


# print(data["address"][0])

# print(data.dtypes)
# print(data)

# #print(data['address_subways'])

