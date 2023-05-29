import pandas as pd
import csv

data = pd.read_csv('ed_data.csv')


for i in range(len(data['published'])):
    data['published'][i] = data['published'][i][:10]

print(data)

print(data.keys())
with open('ed_data_new.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data.keys())
    writer.writeheader()
    print(len(data['published']))
    for i in range(len(data['published'])):
        dict={}
        for key in data.keys():
            dict[key] = data[key][i]
        writer.writerow(dict)