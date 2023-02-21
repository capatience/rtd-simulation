import json
import csv
import pandas as pd

def import_data_mahdi(filename:str) -> pd.DataFrame:
    skiprows = 23
    df = pd.read_csv(filename, skiprows=skiprows)
    df['s'] = df['ms']/1000
    return df

def import_sim_data(filename:str) -> pd.DataFrame:
    df = pd.read_csv(filename, index_col=0)
    return df

def export_df_data(data:pd.DataFrame, filename:str):
    data.to_csv(f"./output/{filename}.csv")
    return

def make_json(csvfilepath, jsonfilepath):
    '''
    Makes a json file from a csv file
    '''
    data = {}
    
    with open(csvfilepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rows in reader:
            key = rows['parameter']
            data[key] = rows

    with open(jsonfilepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4))

def read_json(jsonfilepath):
    data = pd.read_json(jsonfilepath)
    return data

def csv_to_dict(csvfilepath):
    data = {}
    with open(csvfilepath, encoding='utf-8') as f:
        reader = csv.reader(f)

        for row in reader:
            key = row[0]
            try:
                value = float(row[1]) # convert to float if it's a number
                if key.startswith('N'): value = int(row[1])
            except:
                value = row[1]
            data[key] = value
    data.pop('parameter', None)
    return data
