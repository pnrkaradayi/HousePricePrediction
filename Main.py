#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import locale
locale.setlocale(locale.LC_ALL, 'turkish')
import time
from sklearn.utils import shuffle
time.strftime('%c')
from sklearn.model_selection import train_test_split


#veriler_read = pd.read_excel('housepnrdenemeee.xlsx', engine='openpyxl')
veriler_read = pd.read_excel('housepnrdenemeee.xlsx', engine='openpyxl')

veriler_train, veriler_test = train_test_split(veriler_read, test_size=0.2, random_state=42)
veriler_test=veriler_test.reset_index(drop=True)
veriler_train =veriler_train.reset_index(drop=True)
print (len(veriler_test))
print(len(veriler_train))
df_logprice =veriler_train.iloc[:,30:31]
df_features = veriler_train.iloc[:,0:30]
features = veriler_train.iloc[:,0:30].values
logprice =veriler_train.iloc[:,30:31].values
x_train1=df_features
y_train1=df_logprice

df_logprice_test =veriler_test.iloc[:,30:31]
df_features_test = veriler_test.iloc[:,0:31]
features_test = veriler_test.iloc[:,0:31].values
logprice_test =veriler_test.iloc[:,30:31].values
x_test1=df_features_test
y_test1=df_logprice_test
print(x_test1.head(2))

print(np.all(np.isfinite(veriler_read)))
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x_train1, y_train1)
lin_reg_tahmin = lin_reg.predict(x_train1)

df_ytahmin_lr = pd.DataFrame(data=lin_reg_tahmin, index=range(len(y_train1)), columns=["lin_reg_tahmin"])
df_ytt_lr = pd.concat([y_train1, df_ytahmin_lr], axis=1)
df_yttc_lr = df_ytt_lr.corr()
r2_lr = np.float(np.asarray(df_yttc_lr.iloc[0:1, 1:2]) * np.asarray(df_yttc_lr.iloc[0:1, 1:2]))

y_error = y_train1.values - lin_reg_tahmin
df_y_error= pd.DataFrame( data = y_error, index= range(len(y_error)), columns=["y_error"])
veriler_train2 =pd.concat([veriler_train,df_y_error],axis=1)

print(veriler_train2.head(2))
print(len(veriler_train2))

wcss=[]
for i in range(1,10):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=i, random_state=0).fit(df_y_error)
    kmeans.labels_
    #print(kmeans.cluster_centers_)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_y_error)
kmeans.labels_
#print(kmeans.cluster_centers_)
#veriler.append(kmeans.inertia_)
veriler_train2['cluster'] = kmeans.labels_
veriler_train2.head(2)
print(veriler_train2.groupby('cluster')['cluster'].count())

df_logprice_s =veriler_train2.iloc[:,32:33]
df_features_s = veriler_train2.iloc[:,0:31]
features_s = veriler_train2.iloc[:,0:31].values
logprice_s =veriler_train2.iloc[:,32:33].values
x_train_s=df_features_s
y_train_s=df_logprice_s

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train_s,y_train_s)
knn_y_pred = knn.predict(x_test1)

df_knn_y_pred= pd.DataFrame( data = knn_y_pred, index= range(len(knn_y_pred)), columns=["cluster"])
df_knn_test=pd.concat([x_test1,df_knn_y_pred],axis=1)


veriler_all=pd.concat([veriler_train2,df_knn_test],axis=0,ignore_index=True)
veriler = veriler_all[["BsmtFullBath", "GarageQual", "GarageCond", "CentralAir", "GarageType",
                                         "LotArea", "ExterQual", "LotShape", "GarageYrBlt", "HalfBath", "OpenPorchSF",
                                         "2ndFlrSF", "WoodDeckSF", "BsmtFinType1", "BsmtFinSF1", "FireplaceQu",
                                         "HeatingQC", "Foundation", "Fireplaces", "MasVnrArea", "YearRemodAdd",
                                         "YearBuilt", "TotRmsAbvGrd", "FullBath", "1stFlrSF", "TotalBsmtSF",
                                         "GarageArea", "GarageCars", "GrLivArea", "OverallQual", "SalePrice","cluster"]]
veriler.head(2)

veriler.groupby('cluster')['cluster'].count()

import math as math
def mape(actual, predict):
    tmp, n = 0.0, 0
    for i in range(0, len(actual)):
        if actual[i] != 0:
            tmp += math.fabs(actual[i]-predict[i])/actual[i]
            n += 1
    return float((tmp/n)*100)


# Cluster version
import warnings

warnings.filterwarnings("ignore")
# from sklearn.utils import shuffle
from sklearn.model_selection import KFold  # import KFold

# veriler2= veriler.head(100)
veriler2 = veriler.sort_values(by=['cluster'])
cluster_list = veriler.cluster.unique()

fold1 = 0
fold2 = 0

df_mse1 = pd.DataFrame(
    columns=['fold1', 'fold2', 'i', 'c', 'gamma', 'nu', 'mse', 'rmse', 'mae', 'mape', 'r2', 'adjusted_r2_square'])
for i in range(0, len(cluster_list)):
    df_mse2 = pd.DataFrame(columns=['fold1', 'fold2', 'i', 'c', 'gamma', 'nu', 'mse'])
    kfold_cluster = []
    for cls_veriler in veriler2.values:
        if i in cls_veriler[31:32]:
            kfold_cluster.append(cls_veriler)
    X1 = np.asarray(kfold_cluster)
    kf1 = KFold(5, True, 1)
    for train1, test1 in kf1.split(X1):
        fold1 += 1
        train1 = X1[train1]
        test1 = X1[test1]
        df_train1 = pd.DataFrame(data=train1, index=range(len(train1)),
                                 columns=["BsmtFullBath", "GarageQual", "GarageCond", "CentralAir", "GarageType", "LotArea", "ExterQual", "LotShape", "GarageYrBlt", "HalfBath", "OpenPorchSF", "2ndFlrSF", "WoodDeckSF", "BsmtFinType1", "BsmtFinSF1", "FireplaceQu", "HeatingQC", "Foundation", "Fireplaces", "MasVnrArea", "YearRemodAdd", "YearBuilt","TotRmsAbvGrd", "FullBath", "1stFlrSF", "TotalBsmtSF", "GarageArea", "GarageCars", "GrLivArea", "OverallQual", "SalePrice","cluster"])
        df_test1 = pd.DataFrame(data=test1, index=range(len(test1)),
                                columns=["BsmtFullBath", "GarageQual", "GarageCond", "CentralAir", "GarageType",
                                         "LotArea", "ExterQual", "LotShape", "GarageYrBlt", "HalfBath", "OpenPorchSF",
                                         "2ndFlrSF", "WoodDeckSF", "BsmtFinType1", "BsmtFinSF1", "FireplaceQu",
                                         "HeatingQC", "Foundation", "Fireplaces", "MasVnrArea", "YearRemodAdd",
                                         "YearBuilt", "TotRmsAbvGrd", "FullBath", "1stFlrSF", "TotalBsmtSF",
                                         "GarageArea", "GarageCars", "GrLivArea", "OverallQual", "SalePrice","cluster"])

        y_test1 = df_test1.iloc[:, 30:31]
        x_test1 = df_test1.iloc[:, 0:30]
        y_train1 = df_train1.iloc[:, 30:31]
        x_train1 = df_train1.iloc[:, 0:30]

        X2 = df_train1.values
        kf2 = KFold(5, True, 1)
        for train2, test2 in kf2.split(X2):
            fold2 += 1
            train2 = X2[train2]
            test2 = X2[test2]
            df_train2 = pd.DataFrame(data=train2, index=range(len(train2)),
                                     columns=["BsmtFullBath", "GarageQual", "GarageCond", "CentralAir", "GarageType",
                                         "LotArea", "ExterQual", "LotShape", "GarageYrBlt", "HalfBath", "OpenPorchSF",
                                         "2ndFlrSF", "WoodDeckSF", "BsmtFinType1", "BsmtFinSF1", "FireplaceQu",
                                         "HeatingQC", "Foundation", "Fireplaces", "MasVnrArea", "YearRemodAdd",
                                         "YearBuilt", "TotRmsAbvGrd", "FullBath", "1stFlrSF", "TotalBsmtSF",
                                         "GarageArea", "GarageCars", "GrLivArea", "OverallQual", "SalePrice","cluster"])
            df_test2 = pd.DataFrame(data=test2, index=range(len(test2)),
                                    columns=["BsmtFullBath", "GarageQual", "GarageCond", "CentralAir", "GarageType",
                                         "LotArea", "ExterQual", "LotShape", "GarageYrBlt", "HalfBath", "OpenPorchSF",
                                         "2ndFlrSF", "WoodDeckSF", "BsmtFinType1", "BsmtFinSF1", "FireplaceQu",
                                         "HeatingQC", "Foundation", "Fireplaces", "MasVnrArea", "YearRemodAdd",
                                         "YearBuilt", "TotRmsAbvGrd", "FullBath", "1stFlrSF", "TotalBsmtSF",
                                         "GarageArea", "GarageCars", "GrLivArea", "OverallQual", "SalePrice","cluster"])
            y_test2 = df_test2.iloc[:, 30:31]
            x_test2 = df_test2.iloc[:, 0:30]
            y_train2 = df_train2.iloc[:, 30:31]
            x_train2 = df_train2.iloc[:, 0:30]

        #     C2 = [1, 10, 100, 1000]
        #     G2 = [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5]
        #     NU2 = [0.1, 0.3, 0.5, 0.7, 1.0]
        #     for a in range(0, len(C2)):
        #         CC = C2[a]
        #         for b in range(0, len(G2)):
        #             GG = G2[b]
        #             for cn in range(0, len(NU2)):
        #                 NUU = NU2[cn]
        #
        #                 from sklearn.svm import NuSVR
        #
        #                 nusvr2 = NuSVR(nu=NUU, C=CC, kernel='rbf', gamma=GG).fit(x_train2, y_train2)
        #                 nusvr2_tahmin2 = nusvr2.predict(x_test2)
        #                 from sklearn.metrics import mean_squared_error
        #
        #                 nusvr_mse2 = mean_squared_error(y_test2, nusvr2_tahmin2)
        #                 # print(fold1,fold2,"cluster : " ,i, "C: ",CC, "Gamma :", GG,"NU :", NUU, "mse :", nusvr_mse2)
        #                 df_mse2 = df_mse2.append(pd.Series([fold1, fold2, i, CC, GG, NUU, nusvr_mse2],
        #                                                    index=['fold1', 'fold2', 'i', 'c', 'gamma', 'nu', 'mse']),
        #                                          ignore_index=True)
        #                 # df_mse2.style.set_precision(10)
        #
        # xxx = df_mse2.iloc[df_mse2.groupby('i')['mse'].idxmin()]
        # C1 = float(xxx.loc[:, "c"].values)
        # G1 = float(xxx.loc[:, "gamma"].values)
        # NU1 = float(xxx.loc[:, "nu"].values)
        from sklearn.svm import NuSVR
        from sklearn.model_selection import GridSearchCV  # for hypertuning

        ################################################################
#         from sklearn.tree import DecisionTreeRegressor  # for decisiton tree regression
#         nusvr1 = DecisionTreeRegressor(random_state=10,
#                                     max_depth=13,
#                                     min_samples_leaf=4,
#                                     min_samples_split=4,
#                                     max_features='auto')
#
#         nusvr1.fit(x_train1, y_train1)
#         nusvr1_tahmin1 = nusvr1.predict(x_test1)
#         from sklearn.metrics import mean_squared_error
#
#         nusvr1_mse1 = mean_squared_error(y_test1, nusvr1_tahmin1)
#         nusvr1_rmse1 = nusvr1_mse1 ** (1 / 2)
#         from sklearn.metrics import mean_absolute_error
#
#         nusvr1_mae1 = mean_absolute_error(y_test1, nusvr1_tahmin1)
#         nusvr1_mape1 = mape(y_test1.values, nusvr1_tahmin1)
#
#         df_ytahmin = pd.DataFrame(data=nusvr1_tahmin1, index=range(len(nusvr1_tahmin1)), columns=["nusvr1_tahmin1"])
#         df_ytt = pd.concat([y_test1, df_ytahmin], axis=1)
#         df_yttc = df_ytt.corr()
#         r2 = round(np.float(np.asarray(df_yttc.iloc[0:1, 1:2]) * np.asarray(df_yttc.iloc[0:1, 1:2])), 2)
#
#         df_mse1 = df_mse1.append(pd.Series(
#             [fold1, fold2, i, nusvr1_mse1, nusvr1_rmse1, nusvr1_mae1, nusvr1_mape1, r2,
#              round(nusvr1.score(x_test1, y_test1), 2)],
#             index=['fold1', 'fold2', 'i', 'mse', 'rmse', 'mae', 'mape', 'r2',
#                    'adjusted_r2_square']), ignore_index=True)
#
# print(df_mse1.groupby('i')['mse', 'rmse', 'mae', 'mape', 'r2'].mean())
# print(df_mse1.iloc[:,6:11].mean())


        ################################################################
        from sklearn.ensemble import RandomForestRegressor

        # param_grid = {
        #     "n_estimators": [100, 200, 300],
        #     "max_depth": [10, 50, 100],
        #     "max_features": [6, 8, 10, 12, 14, 16]
        # }
        #
        # rf_reg = RandomForestRegressor()
        #
        # nusvr1 = GridSearchCV(estimator=rf_reg,
        #                       param_grid=param_grid,
        #                       cv=2,
        #                       n_jobs=-1,
        #                       verbose=2)
        #
        # nusvr1.fit(x_train1, y_train1)

##################################################################################################################
        nusvr1 = RandomForestRegressor(n_estimators= 100, random_state = 10,oob_score=True,
                           max_depth=7, min_samples_leaf=3,min_samples_split=9,n_jobs=-1)


        nusvr1.fit(x_train1, y_train1)
        nusvr1_tahmin1 = nusvr1.predict(x_test1)
        from sklearn.metrics import mean_squared_error

        nusvr1_mse1 = mean_squared_error(y_test1, nusvr1_tahmin1)
        nusvr1_rmse1 = nusvr1_mse1 ** (1 / 2)
        from sklearn.metrics import mean_absolute_error

        nusvr1_mae1 = mean_absolute_error(y_test1, nusvr1_tahmin1)
        nusvr1_mape1 = mape(y_test1.values, nusvr1_tahmin1)

        df_ytahmin = pd.DataFrame(data=nusvr1_tahmin1, index=range(len(nusvr1_tahmin1)), columns=["nusvr1_tahmin1"])
        df_ytt = pd.concat([y_test1, df_ytahmin], axis=1)
        df_yttc = df_ytt.corr()
        r2 = round(np.float(np.asarray(df_yttc.iloc[0:1, 1:2]) * np.asarray(df_yttc.iloc[0:1, 1:2])), 2)

        df_mse1 = df_mse1.append(pd.Series(
            [fold1, fold2, i, nusvr1_mse1, nusvr1_rmse1, nusvr1_mae1, nusvr1_mape1, r2,
             round(nusvr1.score(x_test1, y_test1), 2)],
            index=['fold1', 'fold2', 'i', 'mse', 'rmse', 'mae', 'mape', 'r2',
                   'adjusted_r2_square']), ignore_index=True)

print(df_mse1.groupby('i')['mse', 'rmse', 'mae', 'mape', 'r2'].mean())
print(df_mse1.iloc[:,6:11].mean())
#
#         from sklearn.ensemble import AdaBoostRegressor
#         nusvr1 = AdaBoostRegressor(random_state=0, n_estimators=100)
#
#
#
#         nusvr1.fit(x_train1, y_train1)
#         nusvr1_tahmin1 = nusvr1.predict(x_test1)
#         from sklearn.metrics import mean_squared_error
#
#         nusvr1_mse1 = mean_squared_error(y_test1, nusvr1_tahmin1)
#         nusvr1_rmse1 = nusvr1_mse1 ** (1 / 2)
#         from sklearn.metrics import mean_absolute_error
#
#         nusvr1_mae1 = mean_absolute_error(y_test1, nusvr1_tahmin1)
#         nusvr1_mape1 = mape(y_test1.values, nusvr1_tahmin1)
#
#         df_ytahmin = pd.DataFrame(data=nusvr1_tahmin1, index=range(len(nusvr1_tahmin1)), columns=["nusvr1_tahmin1"])
#         df_ytt = pd.concat([y_test1, df_ytahmin], axis=1)
#         df_yttc = df_ytt.corr()
#         r2 = round(np.float(np.asarray(df_yttc.iloc[0:1, 1:2]) * np.asarray(df_yttc.iloc[0:1, 1:2])), 2)
#
#         df_mse1 = df_mse1.append(pd.Series(
#             [fold1, fold2, i, nusvr1_mse1, nusvr1_rmse1, nusvr1_mae1, nusvr1_mape1, r2,
#              round(nusvr1.score(x_test1, y_test1), 2)],
#             index=['fold1', 'fold2', 'i', 'mse', 'rmse', 'mae', 'mape', 'r2',
#                    'adjusted_r2_square']), ignore_index=True)
#
# print(df_mse1.groupby('i')['mse', 'rmse', 'mae', 'mape', 'r2'].mean())
# print(df_mse1.iloc[:,6:11].mean())


#############################

#         import xgboost
#         nusvr1 = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
#
#
#
#         nusvr1.fit(x_train1, y_train1)
#         nusvr1_tahmin1 = nusvr1.predict(x_test1)
#         from sklearn.metrics import mean_squared_error
#
#         nusvr1_mse1 = mean_squared_error(y_test1, nusvr1_tahmin1)
#         nusvr1_rmse1 = nusvr1_mse1 ** (1 / 2)
#         from sklearn.metrics import mean_absolute_error
#
#         nusvr1_mae1 = mean_absolute_error(y_test1, nusvr1_tahmin1)
#         nusvr1_mape1 = mape(y_test1.values, nusvr1_tahmin1)
#
#         df_ytahmin = pd.DataFrame(data=nusvr1_tahmin1, index=range(len(nusvr1_tahmin1)), columns=["nusvr1_tahmin1"])
#         df_ytt = pd.concat([y_test1, df_ytahmin], axis=1)
#         df_yttc = df_ytt.corr()
#         r2 = round(np.float(np.asarray(df_yttc.iloc[0:1, 1:2]) * np.asarray(df_yttc.iloc[0:1, 1:2])), 2)
#
#         df_mse1 = df_mse1.append(pd.Series(
#             [fold1, fold2, i, nusvr1_mse1, nusvr1_rmse1, nusvr1_mae1, nusvr1_mape1, r2,
#              round(nusvr1.score(x_test1, y_test1), 2)],
#             index=['fold1', 'fold2', 'i', 'mse', 'rmse', 'mae', 'mape', 'r2',
#                    'adjusted_r2_square']), ignore_index=True)
#
# print(df_mse1.groupby('i')['mse', 'rmse', 'mae', 'mape', 'r2'].mean())
# print(df_mse1.iloc[:,6:11].mean())


