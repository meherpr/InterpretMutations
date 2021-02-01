#The inputs that are required to run the script can be obtained by  typing "python OptimizeparamsPred.py -h"
#An example to run the script is :
#python OptimizeparamsPred.py -i aph3.csv -l 0.01 -f 14

import pandas as pd
import xgboost as xgb
import shap

from sklearn import model_selection
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import argparse
from sys import argv

parser = argparse.ArgumentParser()
parser.add_argument('-l','--lr', dest = "lr",help='learning_rate',type=float, required=True)
parser.add_argument('-f','--colnum', dest= "colnum", help="target column number",type=int,required=True)
parser.add_argument('-i','--input', dest= "infile", help="input file name",type=str,required=True)

options = parser.parse_args()

string = "# python"
for ele in argv:
        string +=  " "+ele
print ("\n",string,"\n")
shap.initjs()

#Load data
X = pd.read_csv(options.infile, header='infer',delimiter=",",usecols=([i for i in range (12)]),index_col=0)
Y = pd.read_csv(options.infile, header='infer',delimiter=",",usecols=[0,options.colnum-1],index_col=0)


lr = options.lr 

test_size = 0.25
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=123)
data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

#### Tuning n_estimators

print ("Tuning n_estimators\n")
xgb1 = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = lr, max_depth=5, min_child_weight=1,gamma=0, subsample=0.8, colsample_bytree=0.8, seed=27)
xgb_param = xgb1.get_xgb_params()
cvresult = xgb.cv(xgb_param, data_dmatrix, num_boost_round=5000, nfold=5, early_stopping_rounds=10)
#print (cvresult.to_string())
n_est_opt = cvresult.index[-1]+1
print ("Optimized n_estimators = ",n_est_opt)


#### Tuning 'max_depth' and 'min_child_weight'

print ("\nTuning 'max_depth' and 'min_child_weight'\n")
param_test2 = {
 'max_depth':[3,5,7,9,15,20,25,50],
 'min_child_weight':[1,3,5,7,9,15,20,25,50]
}

gsearch2 = GridSearchCV(estimator = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = lr, gamma=0, subsample=0.8, colsample_bytree=0.8, seed=27,n_estimators=n_est_opt), param_grid = param_test2, scoring=['neg_mean_squared_error','r2'],cv=5,refit='neg_mean_squared_error')
gsearch2.fit(X_train,y_train)
print ("Updated XGB Parameters",(gsearch2.best_estimator_).get_xgb_params())
#print (pd.DataFrame(gsearch2.cv_results_).to_string())
#print (gsearch2.best_params_, gsearch2.best_score_)
print ("\nOptimized parameter values",gsearch2.best_params_)
maxdepth_opt = gsearch2.best_params_["max_depth"]
min_child_weight_opt = gsearch2.best_params_["min_child_weight"]
#print (maxdepth_opt,min_child_weight_opt,"\n\n")


#### Tuning gamma

print ("\nTuning gamma\n")
param_test3 = {
 'gamma':[i/10.0 for i in range(0,10)]
}

gsearch3 = GridSearchCV(estimator = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate =lr, subsample=0.8, colsample_bytree=0.8, seed=27,n_estimators=n_est_opt,max_depth=maxdepth_opt,min_child_weight=min_child_weight_opt), param_grid = param_test3, scoring=['neg_mean_squared_error','r2'],cv=5,refit='neg_mean_squared_error')

gsearch3.fit(X_train,y_train)
print ("Updated XGB Parameters",(gsearch3.best_estimator_).get_xgb_params())
#print (pd.DataFrame(gsearch3.cv_results_).to_string())
#print (gsearch3.best_params_, gsearch3.best_score_)
print ("\nOptimized parameter values",gsearch3.best_params_)
gamma_opt = gsearch3.best_params_["gamma"]
#print (gamma_opt,"\n\n")


#### Tuning "subsample" and "colsample_bytree"

print ("\nTuning 'subsample' and 'colsample_bytree'\n")
param_test4 = {
 'subsample':[i/10.0 for i in range(4,10)],
 'colsample_bytree':[i/10.0 for i in range(4,10)]
}

gsearch4 = GridSearchCV(estimator = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate =lr, seed=27,n_estimators=n_est_opt,max_depth=maxdepth_opt,min_child_weight=min_child_weight_opt,gamma=gamma_opt), param_grid = param_test4, scoring=['neg_mean_squared_error','r2'],cv=5,refit='neg_mean_squared_error')

gsearch4.fit(X_train,y_train)
print ("Updated XGB Parameters",(gsearch4.best_estimator_).get_xgb_params())
#print (pd.DataFrame(gsearch4.cv_results_).to_string())
print ("\nOptimized parameter values",gsearch4.best_params_)
subsample_opt = gsearch4.best_params_["subsample"]
colsample_bytree_opt = gsearch4.best_params_["colsample_bytree"]
#print (subsample_opt,colsample_bytree_opt,"\n\n")


#### Tuning "reg_alpha"

print ("\nTuning 'reg_alpha'\n")
param_test5 = {
 'reg_alpha':[0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
}
gsearch5 = GridSearchCV(estimator = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate =lr, seed=27,n_estimators=n_est_opt,max_depth=maxdepth_opt,min_child_weight=min_child_weight_opt,gamma=gamma_opt,subsample=subsample_opt, colsample_bytree=colsample_bytree_opt), param_grid = param_test5, scoring=['neg_mean_squared_error','r2'],cv=5,refit='neg_mean_squared_error')

gsearch5.fit(X_train,y_train)
print ("Updated XGB Parameters",(gsearch5.best_estimator_).get_xgb_params())
#print (pd.DataFrame(gsearch5.cv_results_).to_string())
#print (gsearch5.best_params_, gsearch5.best_score_)
print ("\nOptimized parameter values",gsearch5.best_params_)
reg_alpha_opt = gsearch5.best_params_["reg_alpha"]
#print (reg_alpha_opt,"\n\n")

print ("\n======================================\nOptimized parameters for learning rate",lr,"\n======================================\n")
print ("n_estimators =",n_est_opt,"\nmax_depth =",maxdepth_opt,"\nmin_child_weight =",min_child_weight_opt,"\ngamma =",gamma_opt,"\nsubsample =", subsample_opt,"\ncolsample_bytree =", colsample_bytree_opt, "\nreg_alpha =",reg_alpha_opt)

#### Training model with the optimized parameters

print ("Training model with the optimized parameters...\n")
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate =lr, seed=27,n_estimators=n_est_opt,max_depth=maxdepth_opt,min_child_weight=min_child_weight_opt,gamma=gamma_opt,subsample=subsample_opt, colsample_bytree=colsample_bytree_opt,reg_alpha=reg_alpha_opt) 


xg_reg.fit(X_train,y_train)
#print ("Updated XGB Parameters",xg_reg.get_xgb_params())
y_pred = xg_reg.predict(X_test)
y_train_pred = xg_reg.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE: %f" % (rmse))
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Train RMSE: %f" % (rmse))

