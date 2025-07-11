#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sklearn
import argparse
import os

# Argumentos por CLI
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()

year = args.year
month = args.month

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

# Asegurar carpeta
os.makedirs('output', exist_ok=True)

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

# Imprime el resultado
print(f"Mean predicted duration: {y_pred.mean():.2f} minutes")

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
