#!/bin/bash

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

filename = 'trained_model.sav'
condition_map = {'salvage':1, 'fair': 2, 'good': 3, 'excellent':4,'like new':5,'new':6}


def encode(data_frame, column, encoder):
    encoder.fit(data_frame[[column]])

    tmp = pd.DataFrame(data=encoder.transform(data_frame[[column]]), columns=encoder.get_feature_names_out())
    data_frame.drop(columns=[column], axis=1, inplace=True)
    data_frame = pd.concat([data_frame.reset_index(drop=True), tmp], axis=1)
    
    joblib.dump(encoder, column + '_encoder.joblib')

    return data_frame

def encode_pred(data_frame, column, encoder):
    tmp = pd.DataFrame(data=encoder.transform(data_frame[[column]]), columns=encoder.get_feature_names_out())
    data_frame.drop(columns=[column], axis=1, inplace=True)
    data_frame = pd.concat([data_frame.reset_index(drop=True), tmp], axis=1)

    return data_frame

def train():
    # load data
    print('[TRAIN] loading dataset')
    df = pd.read_csv('./vehicles.csv')
    df = df.set_index('id')

    # remove outliers
    df = df[df['price'] < 10000000]
    df = df[df['price'] > 1000]

    df = df[df['manufacturer'] == 'ford']
    df = df[df['price'] < 300000]

    # fix data types
    print('[TRAIN] fixing data types')
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(pd.Int64Dtype())
    df['cylinders'].apply(lambda x: '6 cylinders' if x=='other' else x)
    df['cylinders'] = pd.to_numeric(df['cylinders'].apply(lambda x: str(x).split()[0]), errors='coerce').astype(pd.Int32Dtype())
    df['posting_date'] = df['posting_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None))

    today = datetime.today()
    df['posting_date'] = (today - df['posting_date']).dt.days
    df.dtypes

    # select relevant features
    print('[TRAIN] selecting relevant features')
    df_feat = df.copy()
    df_feat = df_feat[['price','year','condition','cylinders','odometer','fuel', 'title_status', 'transmission','drive','size','type','posting_date']]

    # remove rows with majority NaNs
    nan_cnt = df_feat.isnull().sum(axis=1)
    idx = nan_cnt < 8
    df_feat = df_feat[idx]

    # handle missing values
    print('[TRAIN] handling missing values')
    df_feat = df_feat.dropna(subset=['year'])
    df_feat['cylinders'].fillna(df_feat['cylinders'].median(), inplace=True)
    df_feat['odometer'].fillna(df_feat['odometer'].median(), inplace=True)
    df_feat['fuel'].fillna(df_feat['fuel'].mode()[0], inplace=True)
    df_feat['type'].fillna(df_feat['type'].mode()[0], inplace=True)
    df_feat['title_status'].fillna(df_feat['title_status'].mode()[0], inplace=True)
    df_feat['transmission'].fillna(df_feat['transmission'].mode()[0], inplace=True)
    df_feat['drive'].fillna(df_feat['drive'].mode()[0], inplace=True)
    df_feat.drop(['size'], axis=1, inplace=True)


    # one hot encoding
    fuel_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
    df_feat = encode(df_feat, 'fuel', fuel_encoder)
    
    type_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
    df_feat = encode(df_feat, 'type', type_encoder)

    title_status_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
    df_feat = encode(df_feat, 'title_status', title_status_encoder)

    transmission_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
    df_feat = encode(df_feat, 'transmission', transmission_encoder)

    drive_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='int')
    df_feat = encode(df_feat, 'drive', drive_encoder)

    # condition kNN imputer
    df_feat['condition'] = df_feat['condition'].map(condition_map)

    imputer = KNNImputer(n_neighbors = 3)
    df_feat = pd.DataFrame(imputer.fit_transform(df_feat),columns = df_feat.columns)

    X_train = df_feat.drop(['price'], axis=1)
    y_train = df_feat['price']

    print('[TRAIN] training model')
    model = GradientBoostingRegressor(n_estimators=400, max_depth=10, learning_rate=0.1, random_state=42).fit(X_train,y_train.ravel())

    joblib.dump(model, filename)
    print('[TRAIN] done')


def predict(df):
    filename = 'trained_model.sav'
    model = joblib.load(filename)
    fuel_encoder = joblib.load('fuel_encoder.joblib')
    type_encoder = joblib.load('type_encoder.joblib')
    title_status_encoder = joblib.load('title_status_encoder.joblib')
    transmission_encoder = joblib.load('transmission_encoder.joblib')
    drive_encoder = joblib.load('drive_encoder.joblib')

    df = encode_pred(df, 'fuel', fuel_encoder)
    df = encode_pred(df, 'type', type_encoder)
    df = encode_pred(df, 'title_status', title_status_encoder)
    df = encode_pred(df, 'transmission', transmission_encoder)
    df = encode_pred(df, 'drive', drive_encoder)

    df['condition'] = df['condition'].map(condition_map)

    predictions = model.predict(df)
    return predictions

if __name__ == "__main__":
    train()