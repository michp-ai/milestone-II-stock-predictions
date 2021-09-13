import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Ridge, Lasso
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy


def pred_to_clf_pred(pred, threshold=1):
    pred_clf = (pred > threshold) * 1 # multiply by 1 to change to binary from True / False
    
    return pred_clf

def threshold_precision(y, pred, percentile):
    threshold = np.percentile(pred, percentile)
    pred_clf = (pred > threshold) * 1
    threshold_precision = precision_score(y, pred_clf)
    
    return round(threshold_precision,5)

def full_and_threshold_scoring(y, y_pred, percentile):
    threshold = np.percentile(y_pred, percentile)
    results = {}
    y_clf = pred_to_clf_pred(y, threshold=1)
    y_pred_clf = pred_to_clf_pred(y_pred, threshold=1)
    results['default_precision'] = round(precision_score(y_clf, y_pred_clf),5)
    results['default_return'] = round(np.mean(y[y_pred>1])-1,5)
    results['threshold_precision'] = threshold_precision(y_clf, y_pred, percentile)
    results['threshold_return'] = round(np.mean(y[y_pred>threshold])-1,5)
    
    return results

def feature_engineering(df):
    df['Volume_over_Volume_MA50'] = df['Volume'] / df['Volume_MA50']
    df['Volume_over_Volume_MA200'] = df['Volume'] / df['Volume_MA200']
    df['Volume_MA50_over_Volume_MA200'] = df['Volume_MA50'] / df['Volume_MA200']
    
    return df

def get_numeric_non_infinite_cols(df):
    df = df._get_numeric_data()
    col_has_inf = df.columns.to_series()[np.isinf(df).any()].to_list()
    df = df.drop(col_has_inf, axis=1)
    cols = df.columns.to_list()
    
    return df, cols

def add_pca_cols(df, pca_array):
    df_with_pca = copy.deepcopy(df)
    for m in range(pca_array.shape[1]):
        df_with_pca['pca_' + str(m)] = pca_array[:,m]
        
    return df_with_pca

def scale_train_test(X_tr, X_te):
    scaler = StandardScaler()
    # fit on the train data only
    scaler.fit(X_tr)
    # transform train and test
    X_train_scaled = scaler.transform(X_tr)
    X_test_scaled = scaler.transform(X_te)
    
    return X_train_scaled, X_test_scaled

def pca_train_test(X_train, X_test, num_components=200, random_state=2021):
    pca = PCA(n_components=num_components, random_state=random_state)
    # fit on scaled train data
    pca.fit(X_train)
    # transform scaled train and scaled test
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("Total Explained", sum(pca.explained_variance_ratio_))
    
    return X_train_pca, X_test_pca
