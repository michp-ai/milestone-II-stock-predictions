import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import copy


def sector_mappings(df):
    # utility to clean the sector mappings in the data scraped from pdf's which were incorrect in about 1% of cases 
    # but can be corrected with these mappings
    path = os.path.join('data', "sector_mappings.csv")
    df_sectors = pd.read_csv(path)
    mdict = df_sectors.set_index('Ticker')['Sector'].to_dict()
    df['Sector'] = df['ticker']
    df['Sector'].replace(mdict, inplace=True)
    
    return df

def load_data(return_y_val=False, ohe_sector=True):
    path = os.path.join('data', "milestone_data_X_train.pkl")
    X_train = pd.read_pickle(path)
    X_train = sector_mappings(X_train)    
    path = os.path.join('data', "milestone_data_y_train.pkl")
    y_train = pd.read_pickle(path)
    path = os.path.join('data', "milestone_data_X_test.pkl")
    X_test = pd.read_pickle(path)
    X_test = sector_mappings(X_test)
    path = os.path.join('data', "milestone_data_y_test.pkl")
    y_test = pd.read_pickle(path)
    # validation data - we won't use this until the end
    path = os.path.join('data', "milestone_data_X_val.pkl")
    X_val = pd.read_pickle(path)
    X_val = sector_mappings(X_val)
    
    if ohe_sector:
        X_train = pd.concat([X_train,pd.get_dummies(X_train[['Sector']])],axis=1)
        X_test = pd.concat([X_test,pd.get_dummies(X_test[['Sector']])],axis=1)
        X_val = pd.concat([X_val,pd.get_dummies(X_val[['Sector']])],axis=1)
    
    if return_y_val:
        path = os.path.join('data', "milestone_data_y_val.pkl")
        y_val = pd.read_pickle(path)
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        return X_train, y_train, X_test, y_test, X_val

def pred_to_clf_pred(pred, threshold=1):
    pred_clf = (pred > threshold) * 1 # multiply by 1 to change to binary from True / False
    
    return pred_clf

def threshold_precision(y, pred, percentile):
    threshold = np.percentile(pred, percentile)
    pred_clf = (pred > threshold) * 1
    threshold_precision = precision_score(y, pred_clf)
    
    return round(threshold_precision,4)

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

def feature_engineering(df):
    df['Volume_over_Volume_MA50'] = df['Volume'] / df['Volume_MA50']
    df['Volume_over_Volume_MA200'] = df['Volume'] / df['Volume_MA200']
    df['Volume_MA50_over_Volume_MA200'] = df['Volume_MA50'] / df['Volume_MA200']
    
    
    return df

def threshold_precision(y, pred, percentile):
    threshold = np.percentile(pred, percentile)
    pred_clf = (pred > threshold) * 1
    threshold_precision = precision_score(y, pred_clf)
    
    return round(threshold_precision,4)

def full_and_threshold_scoring(y, y_pred, percentile):
    threshold = np.percentile(y_pred, percentile)
    results = {}
    y_clf = pred_to_clf_pred(y, threshold=1)
    y_pred_clf = pred_to_clf_pred(y_pred, threshold=1)
    results['default_precision'] = round(precision_score(y_clf, y_pred_clf),4)
    results['default_return'] = round(np.mean(y[y_pred>1])-1,4)
    results['threshold_precision'] = threshold_precision(y_clf, y_pred, percentile)
    results['threshold_return'] = round(np.mean(y[y_pred>threshold])-1,4)
    
    return results

def pred_to_clf_pred(pred, threshold=1):
    pred_clf = (pred > threshold) * 1 # multiply by 1 to change to binary from True / False
    
    return pred_clf

def generate_benchmark_scores(y_train, y_test, y_train_clf, y_all_up_train, y_test_clf, y_all_up_test):
    benchmark_scores = {}
    benchmark_scores['train_precision'] = round(precision_score(y_train_clf, y_all_up_train),4)
    benchmark_scores['test_precision'] = round(precision_score(y_test_clf, y_all_up_test),4)
    benchmark_scores['train_return'] = round(np.mean(y_train) - 1,4)
    benchmark_scores['test_return'] = round(np.mean(y_test) - 1,4)
    
    return benchmark_scores

def return_results(benchmark_scores, train_scores, test_scores):
    train_precision = {}
    train_precision['dataset'] = "train"
    train_precision['metric'] = "precision"
    train_precision['benchmark'] = "{:.3f}".format(benchmark_scores['train_precision'])
    train_precision['default'] = "{:.3f}".format(train_scores['default_precision'])
    train_precision['threshold'] = "{:.3f}".format(train_scores['threshold_precision'])
    test_precision = {}
    test_precision['dataset'] = "test"
    test_precision['metric'] = "precision"
    test_precision['benchmark'] = "{:.3f}".format(benchmark_scores['test_precision'])
    test_precision['default'] = "{:.3f}".format(test_scores['default_precision'])
    test_precision['threshold'] = "{:.3f}".format(test_scores['threshold_precision'])
    train_return = {}
    train_return['dataset'] = "train"
    train_return['metric'] = "return"
    train_return['benchmark'] = "{:.2%}".format(benchmark_scores['train_return'])
    train_return['default'] = "{:.2%}".format(train_scores['default_return'])
    train_return['threshold'] = "{:.2%}".format(train_scores['threshold_return'])
    test_return = {}
    test_return['dataset'] = "test"
    test_return['metric'] = "return"
    test_return['benchmark'] = "{:.2%}".format(benchmark_scores['test_return'])
    test_return['default'] = "{:.2%}".format(test_scores['default_return'])
    test_return['threshold'] = "{:.2%}".format(test_scores['threshold_return'])
    results = pd.DataFrame.from_dict(train_precision, orient='index').T
    results = results.append(pd.DataFrame.from_dict(test_precision, orient='index').T)
    results = results.append(pd.DataFrame.from_dict(train_return, orient='index').T)
    results = results.append(pd.DataFrame.from_dict(test_return, orient='index').T)
    results.reset_index(inplace=True)
    results = results.drop(['index'], axis=1)
    
    return results

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

def run_model(model, current_selection, X_tr, X_te, y_train, y_test, model_type="regression", percentile=95):
    np.random.seed(0)
    if model_type=="regression":
        model.fit(X_tr[current_selection], y_train)
        # make the predictions
        y_pred_fundamental_train = model.predict(X_tr[current_selection])
        y_pred_fundamental_test = model.predict(X_te[current_selection])
        train_scores = full_and_threshold_scoring(y_train,y_pred_fundamental_train,percentile)
        test_scores = full_and_threshold_scoring(y_test,y_pred_fundamental_test,percentile)
    elif model_type=="classification":
        train_scores = {}
        test_scores = {}
        y_train_clf = pred_to_clf_pred(y_train, threshold=1)
        y_test_clf = pred_to_clf_pred(y_test, threshold=1)
        model.fit(X_tr[current_selection], y_train_clf)
        # make the predictions
        y_pred_fundamental_train = model.predict_proba(X_tr[current_selection])[:,1]
        y_pred_fundamental_test = model.predict_proba(X_te[current_selection])[:,1]
        # convert to classification
        y_pred_clf_fundamental_train = model.predict(X_tr[current_selection])
        y_pred_clf_fundamental_test = model.predict(X_te[current_selection])
        
        # calculate the return scores
        train_scores['default_precision'] = round(precision_score(y_train_clf, y_pred_clf_fundamental_train),4)
        test_scores['default_precision'] = round(precision_score(y_test_clf, y_pred_clf_fundamental_test),4)
        train_scores['default_return'] = round(np.mean(y_train[y_pred_fundamental_train>0.5])-1,4)
        test_scores['default_return'] = round(np.mean(y_test[y_pred_fundamental_test>0.5])-1,4)
        train_scores['threshold_precision'] = threshold_precision(y_train_clf, y_pred_fundamental_train, percentile)
        test_scores['threshold_precision'] = threshold_precision(y_test_clf, y_pred_fundamental_test, percentile)
        
        # calculate the threshold for the top 5% then calculate the threshold return. Then return the return along with the other scores!
        threshold = np.percentile(y_pred_fundamental_train, percentile)        
        train_scores['threshold_return'] = round(np.mean(y_train[y_pred_fundamental_train>threshold])-1,4)
        threshold = np.percentile(y_pred_fundamental_test, percentile)        
        test_scores['threshold_return'] = round(np.mean(y_test[y_pred_fundamental_test>threshold])-1,4)
        
    return train_scores, test_scores, model
