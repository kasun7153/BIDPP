#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

dir = 'Data/'
processed_data_dir = 'processed_data'


def read_file(feature):
    predictor = feature
    paac_train = pd.read_csv(f'{dir+processed_data_dir}/TR_{feature}.csv')
    paac_test = pd.read_csv(f'{dir+processed_data_dir}/TS_{feature}.csv')
    return predictor, paac_train, paac_test


def balance(df):
    conditions = [
        (df['label'] == 0),
        (df['label'] == 1)
    ]

    values = [0, 1]

    outcomes = np.select(conditions, values)

    rov = RandomOverSampler(random_state=3)
    df_bal, out_bal = rov.fit_resample(df, outcomes)
    return df_bal


def feature_importance(X, Y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, Y)

    importances = rf.feature_importances_

    sorted_idx = importances.argsort()[::-1]
    k = int((70/100)*len(sorted_idx))
    top_k_idx = sorted_idx[:k]
    X_top_k = X.iloc[:, top_k_idx]
    return X_top_k


class BinaryClassifier:
    def __init__(self, models):
        self.models = models
        self.best_model = None

    def train(self, X_train, y_train):
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model['model'].fit(X_train, y_train)
            print(f"{model_name} training completed.")

    def grid_search(self, X_train, y_train, param_grid, cv=5):
        for model_name, model in self.models.items():
            print(f"Performing Grid Search for {model_name}...")
            grid_search = GridSearchCV(
                model['model'], param_grid[model_name], cv=cv, scoring='f1')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            print(f"Best parameters for {model_name}: {best_params}")
            print(f"Best score for {model_name}: {best_score}")
            self.models[model_name]['best_param'] = best_params
            self.models[model_name]['best_score'] = best_score
            self.models[model_name]['model'] = grid_search.best_estimator_
            if self.best_model is None or grid_search.best_score_ > self.best_model['score']:
                self.best_model = {'model_name': model_name, 'model': grid_search.best_estimator_,
                                   'score': grid_search.best_score_, 'params': best_params}

    def get_best_model(self):
        return self.best_model


def report(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    sensitivity = recall_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    precision = precision_score(y_true, y_pred)

    weights = {'Accuracy': 0.3, 'F1': 0.2, 'Sensitivity': 0.2,
               'Specificity': 0.1, 'Precision': 0.2}

    data = {'Accuracy': [accuracy],
            'F1': [f1],
            'Sensitivity': [sensitivity],
            'Specificity': [specificity],
            'Precision': [precision]}
    df = pd.DataFrame(data)

    combined_score = sum([weights[metric] * df[metric].values[0]
                         for metric in weights])
    df['Combined Score'] = combined_score

    print(df)
    df.to_csv(f'out/classification_metrics.csv', index=False)


def output_results(predict_results):
    results = pd.DataFrame({'predicted_values': predict_results})
    results = pd.concat([paac_test['id'], results], axis=1)

    results['Numeric'] = results['id'].str.split('_').str[1].astype(int)

    predictions_neg = results[results['id'].str.startswith('N')]
    predictions_neg = predictions_neg.sort_values('Numeric')

    predictions_neg = predictions_neg.drop('Numeric', axis=1)

    predictions_pos = results[results['id'].str.startswith('P')]

    predictions_pos = predictions_pos.sort_values('Numeric')


    predictions_pos = predictions_pos.drop('Numeric', axis=1)

    predictions_pos.to_csv(f'out/predictions_pos.txt',
                           index=False, header=True, sep='\t')
    predictions_neg.to_csv(f'out/predictions_neg.txt',
                           index=False, header=True, sep='\t')


def run():

    global predictor, paac_train, paac_test
  
    predictor, paac_train, paac_test = read_file('AAC')


    paac_train_bal = balance(paac_train)

    X = paac_train.drop(['id', 'label'], axis=1, inplace=False)
    Y = paac_train['label']


    X_top_k = feature_importance(X, Y)
    preprocessed = pd.concat([paac_train['label'], X_top_k], axis=1)
    preprocessed = pd.concat([paac_train['id'], preprocessed], axis=1)

    X = preprocessed.drop(['id', 'label'], axis=1, inplace=False)
    Y = preprocessed['label']


    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=2)

    important_cols = X.columns


    scaler = StandardScaler()
    normal = ColumnTransformer(
        [('normalize', scaler, important_cols)], remainder='passthrough')
    X_train = pd.DataFrame(normal.fit_transform(
        X_train), columns=important_cols)
    X_val = pd.DataFrame(normal.transform(X_val), columns=important_cols)

    joblib.dump(normal, f'out/{predictor}_normal.joblib')

    models = {
        'SVM': {
            'model': SVC(C=1, kernel='rbf'),
            'best_param': None,
            'best_score': None
        },
        'CatBoost': {
            'model': CatBoostClassifier(),
            'best_param': None,
            'best_score': None
        }
    }


    pipe = Pipeline([('scaler', normal), ('model', models['SVM']['model'])])
    pipe.fit(X_train, Y_train)
    joblib.dump(pipe, f"out/{predictor}_model.joblib")

    predictor_model = joblib.load(f"out/{predictor}_model.joblib")
    normal = joblib.load(f'out/{predictor}_normal.joblib')

    features = paac_test.drop(['id', 'label'], axis=1, inplace=False)
    scaled_features = pd.DataFrame(
        normal.transform(features), columns=important_cols)
    predict_results = pipe.predict(scaled_features)
    report(paac_test.label, predict_results)

 
    output_results(predict_results)
