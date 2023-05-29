import pandas as pd
import csv

data = pd.read_csv('/home/maryna/dynamic-pricing/models/data/keyratedata.csv')


for i in range(len(data['timestamp'])):
    day = data['timestamp'][i][:2]
    month = data['timestamp'][i][3:5]
    year = data['timestamp'][i][6:]
    data['timestamp'][i] = str(year)+"-"+str(month)+"-"+day

print(data)

print(data.keys())
with open('/home/maryna/dynamic-pricing/models/data/keyratedata_new.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data.keys())
    writer.writeheader()
    print(len(data['timestamp']))
    for i in range(len(data['timestamp'])):
        dict={}
        for key in data.keys():
            dict[key] = data[key][i]
        writer.writerow(dict)