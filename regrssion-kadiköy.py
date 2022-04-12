#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
import locale
locale.setlocale(locale.LC_ALL, 'turkish')
import time
from sklearn.utils import shuffle
time.strftime('%c')
from sklearn.tree import DecisionTreeRegressor # for decisiton tree regression
from sklearn.linear_model import LinearRegression
veriler = pd.read_excel('konutozlem2.xlsx', engine='openpyxl')

#veriler = pd.read_excel('konutozlem2.xlsx', engine='openpyxl')
veriler3 = veriler
index = 0
fold1 = 0

import math as math
def mape(actual, predict):
    tmp, n = 0.0, 0
    for i in range(0, len(actual)):
        if actual[i] != 0:
            tmp += math.fabs(actual[i]-predict[i])/actual[i]
            n += 1
    return float((tmp/n)*100)

df_sonuc_nc = pd.DataFrame(
    columns=['index', 'fold1', 'fold2', 'model', 'alpha', 'mse', 'rmse', 'mae', 'mape', 'r2', 'adjusted_r2_score',
             'intercept', 'coef'])

X1 = veriler3.values
kf1 = KFold(5, True, 1)
for train1, test1 in kf1.split(X1):
    df_alpha = pd.DataFrame(columns=['alpha', 'mean_squared_error'])
    df_alpha2 = pd.DataFrame(columns=['alpha', 'mean_squared_error'])
    fold1 += 1
    train1 = X1[train1]
    test1 = X1[test1]
    df_train1 = pd.DataFrame(data=train1, index=range(len(train1)),
                                 columns=["logm2", "nroom", "nbathr_tr", "floor", "nfloor", "insite", "year","logprice"])
    df_test1 =pd.DataFrame(data=test1, index=range(len(test1)),
                                columns=["logm2", "nroom", "nbathr_tr", "floor", "nfloor", "insite", "year", "logprice"])
    y_test1 = df_test1.iloc[:,7:8]
    x_test1 = df_test1.iloc[:,0:7]
    y_train1 = df_train1.iloc[:,7:8]
    x_train1 = df_train1.iloc[:,0:7]

    fold2 = 0
    X2 = df_train1.values
    kf2 = KFold(5, True, 1)
    for train2, test2 in kf2.split(X2):
        fold2 += 1
        train2 = X2[train2]
        test2 = X2[test2]
        df_train2 =  pd.DataFrame(data=train2, index=range(len(train2)),
                                     columns=["logm2", "nroom", "nbathr_tr", "floor", "nfloor", "insite", "year",
                                              "logprice"])
        df_test2 = pd.DataFrame(data=test2, index=range(len(test2)),
                                    columns=["logm2", "nroom", "nbathr_tr", "floor", "nfloor", "insite", "year",
                                             "logprice"])

        y_test2 = df_test2.iloc[:, 7:8]
        x_test2 = df_test2.iloc[:, 0:7]
        y_train2 = df_train2.iloc[:, 7:8]
        x_train2 = df_train2.iloc[:, 0:7]
        # Ridge
        from sklearn.linear_model import Ridge, RidgeCV

        alphasr = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]
        RidgeCV_model2 = RidgeCV(cv=5, alphas=alphasr).fit(x_train2, y_train2)
        alpha2 = RidgeCV_model2.alpha_
        ridge_reg2 = Ridge(alpha=alpha2)
        model3 = 'ridge_reg1'

        ridge_reg2.fit(x_train2, y_train2)
        ridge_reg_tahmin2 = ridge_reg2.predict(x_test2)
        from sklearn.metrics import mean_squared_error

        mean_squared_error3 = mean_squared_error(y_test2, ridge_reg_tahmin2)
        std = np.std(y_test2.values)

        # print(index,fold1,fold2,model1,max_alpha,mean_squared_error1)

        df_alpha = df_alpha.append(pd.Series([alpha2, mean_squared_error3],
                                             index=['alpha', 'mean_squared_error']), ignore_index=True)
        max_alpha = [float(df_alpha.iloc[df_alpha['mean_squared_error']
                           .idxmin():df_alpha['mean_squared_error'].idxmin() + 1, 0:1].values)]

        """     
        #Lasso
        from sklearn.linear_model import Lasso,LassoCV
        alphasr2=[0.0001,0.001,0.01,0.1,0.5,1,10,100,1000]
        LassoCV_model2 = LassoCV(cv=5,max_iter=1000,alphas=alphasr2 ).fit(x_train2, y_train2)
        alpha3= LassoCV_model2.alpha_

        lasso_reg2 = Lasso(alpha=alpha3)
        model4='lasso_reg2'
        lasso_reg2.fit(x_train2, y_train2)
        lasso_reg_tahmin2 = lasso_reg2.predict(x_test2)

        from sklearn.metrics import mean_squared_error 
        mean_squared_error4= mean_squared_error(y_test2, lasso_reg_tahmin2 )
        std2 = np.std(y_test2.values) 

        df_alpha2 = df_alpha2.append(pd.Series([alpha3,mean_squared_error4], 
                                               index=['alpha','mean_squared_error']),ignore_index=True)
        max_alpha2=[float(df_alpha2.iloc[df_alpha2['mean_squared_error']
                                         .idxmin():df_alpha2['mean_squared_error'].idxmin()+1,0:1].values)]
        """

    from sklearn.linear_model import Ridge

    # alphasr2=[0.001,0.01,0.1,1,10,100,1000]
    # max_alpha=[0.001]

    ridge_reg1 = Ridge(alpha=max_alpha)
    model1 = 'ridge_reg1'

    ridge_reg1.fit(x_train1, y_train1)
    ridge_reg_tahmin1 = ridge_reg1.predict(x_test1)
    from sklearn.metrics import mean_squared_error

    mean_squared_error1 = mean_squared_error(y_test1, ridge_reg_tahmin1)

    ridge_reg1_rmse1 = mean_squared_error1 ** (1 / 2)
    from sklearn.metrics import mean_absolute_error

    ridge_reg1_mae1 = mean_absolute_error(y_test1, ridge_reg_tahmin1)
    ridge_reg1_mape1 = mape(y_test1.values, ridge_reg_tahmin1)

    df_ytahmin_rid = pd.DataFrame(data=ridge_reg_tahmin1, index=range(len(ridge_reg_tahmin1)),
                                  columns=["ridge_reg_tahmin1"])
    df_ytt_rid = pd.concat([y_test1, df_ytahmin_rid], axis=1)
    df_yttc_rid = df_ytt_rid.corr()
    r2_rid = np.float(np.asarray(df_yttc_rid.iloc[0:1, 1:2]) * np.asarray(df_yttc_rid.iloc[0:1, 1:2]))

    std = np.std(train1)
    # print(index,fold1,fold2,model1,max_alpha,mean_squared_error1)
    df_sonuc_nc = df_sonuc_nc.append(pd.Series([index, fold1, fold2, model1, max_alpha, mean_squared_error1,
                                                ridge_reg1_rmse1, ridge_reg1_mae1, ridge_reg1_mape1, r2_rid,
                                                ridge_reg1.score(x_test1, y_test1), ridge_reg1.intercept_,
                                                ridge_reg1.coef_],
                                               index=['index', 'fold1', 'fold2', 'model', 'alpha', 'mse', 'rmse', 'mae',
                                                      'mape', 'r2', 'adjusted_r2_score', 'intercept', 'coef']),
                                     ignore_index=True)

    """  
    #Lasso
    from sklearn.linear_model import Lasso
    #alphasr2=[0.001,0.01,0.1,1,10,100,1000]
    #max_alpha2=1000
    lasso_reg1 = Lasso(alpha=max_alpha2)
    model2='lasso_reg1'
    lasso_reg1.fit(x_train1, y_train1)
    lasso_reg_tahmin1 = lasso_reg1.predict(x_test1)

    from sklearn.metrics import mean_squared_error 
    mean_squared_error2= mean_squared_error(y_test1, lasso_reg_tahmin1 )

    df_ytahmin_las= pd.DataFrame( data = lasso_reg_tahmin1, index= range(len(lasso_reg_tahmin1)), columns=["lasso_reg_tahmin1"])
    df_ytt_las=pd.concat([y_test1,df_ytahmin_las],axis=1)
    df_yttc_las=df_ytt_las.corr()
    r2_las = round(np.float(np.asarray(df_yttc_las.iloc[0:1,1:2])*np.asarray(df_yttc_las.iloc[0:1,1:2])),2)

    std2 = np.std(y_test1.values) 
    #print(index,fold1,fold2,model2,max_alpha2,mean_squared_error2)
    df_sonuc = df_sonuc.append(pd.Series([index,fold1,fold2,model2,max_alpha2,mean_squared_error2,r2_las,lasso_reg1.score(x_test1, y_test1),lasso_reg1.intercept_,lasso_reg1.coef_], 
                                             index=['index','fold1','fold2', 'model','alpha', 'mse','r2','score','intercept','coef']), ignore_index=True)
    """

    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    model5 = 'LinearRegression'
    lin_reg.fit(x_train1, y_train1)
    lin_reg_tahmin = lin_reg.predict(x_test1)

    from sklearn.metrics import mean_squared_error

    mean_squared_error5 = mean_squared_error(y_test1, lin_reg_tahmin)

    lin_reg_rmse1 = mean_squared_error5 ** (1 / 2)
    from sklearn.metrics import mean_absolute_error

    lin_reg_mae1 = mean_absolute_error(y_test1, lin_reg_tahmin)
    lin_reg_mape1 = mape(y_test1.values, lin_reg_tahmin)

    df_ytahmin_lr = pd.DataFrame(data=lin_reg_tahmin, index=range(len(lin_reg_tahmin)), columns=["lin_reg_tahmin"])
    df_ytt_lr = pd.concat([y_test1, df_ytahmin_lr], axis=1)
    df_yttc_lr = df_ytt_lr.corr()
    r2_lr = np.float(np.asarray(df_yttc_lr.iloc[0:1, 1:2]) * np.asarray(df_yttc_lr.iloc[0:1, 1:2]))

    std5 = np.std(y_test1.values)
    df_sonuc_nc = df_sonuc_nc.append(
        pd.Series([index, fold1, fold2, model5, 0, mean_squared_error5, lin_reg_rmse1, lin_reg_mae1, lin_reg_mape1,
                   r2_lr, lin_reg.score(x_test1, y_test1), lin_reg.intercept_, lin_reg.coef_],
                  index=['index', 'fold1', 'fold2', 'model', 'alpha', 'mse', 'rmse', 'mae', 'mape', 'r2',
                         'adjusted_r2_score', 'intercept', 'coef']), ignore_index=True)

#print(df_sonuc_nc.groupby(['model'])['mse','rmse','mae','mape','r2'].mean())


#######################################


    dt_reg = DecisionTreeRegressor(random_state=10,
                                    max_depth=13,
                                    min_samples_leaf=4,
                                    min_samples_split=4,
                                    max_features='auto')
    model5 = 'DecisionTreeRegressor'
    dt_reg.fit(x_train1, y_train1)
    dt_reg_tahmin = dt_reg.predict(x_test1)

    from sklearn.metrics import mean_squared_error

    mean_squared_error5 = mean_squared_error(y_test1, dt_reg_tahmin)

    dt_reg_rmse1 = mean_squared_error5 ** (1 / 2)
    from sklearn.metrics import mean_absolute_error

    dt_reg_mae1 = mean_absolute_error(y_test1, dt_reg_tahmin)
    dt_reg_mape1 = mape(y_test1.values, dt_reg_tahmin)

    df_ytahmin_lr = pd.DataFrame(data=dt_reg_tahmin, index=range(len(dt_reg_tahmin)), columns=["dt_reg_tahmin"])
    df_ytt_lr = pd.concat([y_test1, df_ytahmin_lr], axis=1)
    df_yttc_lr = df_ytt_lr.corr()
    r2_lr = np.float(np.asarray(df_yttc_lr.iloc[0:1, 1:2]) * np.asarray(df_yttc_lr.iloc[0:1, 1:2]))

    std5 = np.std(y_test1.values)
    df_sonuc_nc = df_sonuc_nc.append(
        pd.Series([index, fold1, fold2, model5, 0, mean_squared_error5, dt_reg_rmse1, dt_reg_mae1, dt_reg_mape1,
                   r2_lr, dt_reg.score(x_test1, y_test1)],
                  index=['index', 'fold1', 'fold2', 'model', 'alpha', 'mse', 'rmse', 'mae', 'mape', 'r2',
                         'adjusted_r2_score']), ignore_index=True)



####################################

    from sklearn.ensemble import RandomForestRegressor
    rf_reg = RandomForestRegressor()
    model5 = 'RandomForestRegressor'
    rf_reg.fit(x_train1, y_train1)
    rf_reg_tahmin = rf_reg.predict(x_test1)

    from sklearn.metrics import mean_squared_error

    mean_squared_error5 = mean_squared_error(y_test1, rf_reg_tahmin)

    rf_reg_rmse1 = mean_squared_error5 ** (1 / 2)
    from sklearn.metrics import mean_absolute_error

    rf_reg_mae1 = mean_absolute_error(y_test1, rf_reg_tahmin)
    rf_reg_mape1 = mape(y_test1.values, rf_reg_tahmin)

    df_ytahmin_lr = pd.DataFrame(data=rf_reg_tahmin, index=range(len(rf_reg_tahmin)), columns=["rf_reg_tahmin"])
    df_ytt_lr = pd.concat([y_test1, df_ytahmin_lr], axis=1)
    df_yttc_lr = df_ytt_lr.corr()
    r2_lr = np.float(np.asarray(df_yttc_lr.iloc[0:1, 1:2]) * np.asarray(df_yttc_lr.iloc[0:1, 1:2]))

    std5 = np.std(y_test1.values)
    df_sonuc_nc = df_sonuc_nc.append(
        pd.Series([index, fold1, fold2, model5, 0, mean_squared_error5, rf_reg_rmse1, rf_reg_mae1, rf_reg_mape1,
                   r2_lr, rf_reg.score(x_test1, y_test1)],
                  index=['index', 'fold1', 'fold2', 'model', 'alpha', 'mse', 'rmse', 'mae', 'mape', 'r2',
                         'adjusted_r2_score']), ignore_index=True)


####################################

    from sklearn.ensemble import AdaBoostRegressor
    ada_reg = AdaBoostRegressor(random_state=0, n_estimators=100)
    model5 = 'AdaBoostRegressor'
    ada_reg.fit(x_train1, y_train1)
    ada_reg_tahmin = ada_reg.predict(x_test1)

    from sklearn.metrics import mean_squared_error

    mean_squared_error5 = mean_squared_error(y_test1, ada_reg_tahmin)

    ada_reg_rmse1 = mean_squared_error5 ** (1 / 2)
    from sklearn.metrics import mean_absolute_error

    ada_reg_mae1 = mean_absolute_error(y_test1, ada_reg_tahmin)
    ada_reg_mape1 = mape(y_test1.values, ada_reg_tahmin)

    df_ytahmin_lr = pd.DataFrame(data=ada_reg_tahmin, index=range(len(ada_reg_tahmin)), columns=["ada_reg_tahmin"])
    df_ytt_lr = pd.concat([y_test1, df_ytahmin_lr], axis=1)
    df_yttc_lr = df_ytt_lr.corr()
    r2_lr = np.float(np.asarray(df_yttc_lr.iloc[0:1, 1:2]) * np.asarray(df_yttc_lr.iloc[0:1, 1:2]))

    std5 = np.std(y_test1.values)
    df_sonuc_nc = df_sonuc_nc.append(
        pd.Series([index, fold1, fold2, model5, 0, mean_squared_error5, ada_reg_rmse1, ada_reg_mae1, ada_reg_mape1,
                   r2_lr, ada_reg.score(x_test1, y_test1)],
                  index=['index', 'fold1', 'fold2', 'model', 'alpha', 'mse', 'rmse', 'mae', 'mape', 'r2',
                         'adjusted_r2_score']), ignore_index=True)

################################################
    ####################################

    from sklearn.ensemble import AdaBoostRegressor
    import xgboost
    XG_reg = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.2, colsample_bytree=0.8)
    model5 = 'XGBRegressor'
    XG_reg.fit(x_train1, y_train1)
    XG_reg_tahmin = XG_reg.predict(x_test1)

    from sklearn.metrics import mean_squared_error

    mean_squared_error5 = mean_squared_error(y_test1, XG_reg_tahmin)

    XG_reg_rmse1 = mean_squared_error5 ** (1 / 2)
    from sklearn.metrics import mean_absolute_error

    XG_reg_mae1 = mean_absolute_error(y_test1, XG_reg_tahmin)
    XG_reg_mape1 = mape(y_test1.values, XG_reg_tahmin)

    df_ytahmin_lr = pd.DataFrame(data=XG_reg_tahmin, index=range(len(XG_reg_tahmin)), columns=["XG_reg_tahmin"])
    df_ytt_lr = pd.concat([y_test1, df_ytahmin_lr], axis=1)
    df_yttc_lr = df_ytt_lr.corr()
    r2_lr = np.float(np.asarray(df_yttc_lr.iloc[0:1, 1:2]) * np.asarray(df_yttc_lr.iloc[0:1, 1:2]))

    std5 = np.std(y_test1.values)
    df_sonuc_nc = df_sonuc_nc.append(
        pd.Series([index, fold1, fold2, model5, 0, mean_squared_error5, XG_reg_rmse1, XG_reg_mae1, XG_reg_mape1,
                   r2_lr, XG_reg.score(x_test1, y_test1)],
                  index=['index', 'fold1', 'fold2', 'model', 'alpha', 'mse', 'rmse', 'mae', 'mape', 'r2',
                         'adjusted_r2_score']), ignore_index=True)




print(df_sonuc_nc.groupby(['model'])['mse','rmse','mae','mape','r2'].mean())
