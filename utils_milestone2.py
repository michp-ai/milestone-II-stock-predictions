import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression
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

def feature_engineering(df):
    df['Volume_over_Volume_MA50'] = df['Volume'] / df['Volume_MA50']
    df['Volume_over_Volume_MA200'] = df['Volume'] / df['Volume_MA200']
    df['Volume_MA50_over_Volume_MA200'] = df['Volume_MA50'] / df['Volume_MA200']
    
    return df

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

def pred_to_clf_pred(pred, threshold=1):
    pred_clf = (pred > threshold) * 1 # multiply by 1 to change to binary from True / False
    
    return pred_clf

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

def run_model(current_selection, X_tr, X_te, y_train, y_test, model_type="regression", percentile=95):
    np.random.seed(0)
    if model_type=="regression":
        #model = RandomForestRegressor(max_depth=3, random_state=6, criterion="mse", n_jobs=-1) #, min_impurity_decrease=0.01) # 
        model = LinearRegression()
        model.fit(X_tr[current_selection], y_train)
        # make the predictions
        y_pred_fundamental_train = model.predict(X_tr[current_selection])
        y_pred_fundamental_test = model.predict(X_te[current_selection])
        # convert to classification
        y_pred_clf_fundamental_train = pred_to_clf_pred(y_pred_fundamental_train, threshold=1)
        y_pred_clf_fundamental_test = pred_to_clf_pred(y_pred_fundamental_test, threshold=1)
        train_scores = full_and_threshold_scoring(y_train,y_pred_fundamental_train,percentile)
        test_scores = full_and_threshold_scoring(y_test,y_pred_fundamental_test,percentile)
        #print(c, test_scores['threshold_return'] )
    elif model_type=="classification":
        train_scores = {}
        test_scores = {}
        model = LogisticRegression(n_jobs=-1)
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
        train_scores['default_precision'] = round(precision_score(y_train_clf, y_pred_clf_fundamental_train),5)
        test_scores['default_precision'] = round(precision_score(y_test_clf, y_pred_clf_fundamental_test),5)
        train_scores['default_return'] = round(np.mean(y_train[y_pred_fundamental_train>0.5])-1,5)
        test_scores['default_return'] = round(np.mean(y_test[y_pred_fundamental_test>0.5])-1,5)
        train_scores['threshold_precision'] = threshold_precision(y_train_clf, y_pred_fundamental_train, percentile)
        test_scores['threshold_precision'] = threshold_precision(y_test_clf, y_pred_fundamental_test, percentile)
        
        # calculate the threshold for the top 5% then calculate the threshold return. Then return the return along with the other scores!
        threshold = np.percentile(y_pred_fundamental_train, percentile)        
        train_scores['threshold_return'] = round(np.mean(y_train[y_pred_fundamental_train>threshold])-1,5)
        threshold = np.percentile(y_pred_fundamental_test, percentile)        
        test_scores['threshold_return'] = round(np.mean(y_test[y_pred_fundamental_test>threshold])-1,5)
        
    return train_scores, test_scores
