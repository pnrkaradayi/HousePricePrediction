import pandas as pd
import numpy as np
import numpy as np # linear algebra

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



print(train.head())
# string label to categorical values
from sklearn.preprocessing import LabelEncoder

for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

print(train['SaleCondition'].unique())
# Which columns have nan?
print('training data+++++++++++++++++++++')
for i in np.arange(train.shape[1]):
    n = train.iloc[:,i].isnull().sum()
    if n > 0:
        print(list(train.columns.values)[i] + ': ' + str(n) + ' nans')

print('testing data++++++++++++++++++++++ ')
for i in np.arange(test.shape[1]):
    n = test.iloc[:,i].isnull().sum()
    if n > 0:
        print(list(test.columns.values)[i] + ': ' + str(n) + ' nans')


# keep ID for submission
train_ID = train['Id']
test_ID = test['Id']

# split data for training
y_train = train['SalePrice']
X_train = train.drop(['Id','SalePrice'], axis=1)
X_test = test.drop('Id', axis=1)

# dealing with missing data
Xmat = pd.concat([X_train, X_test])
Xmat = Xmat.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)
#Xmat = Xmat.fillna(Xmat.median())
Xmat = Xmat.dropna(how='any',axis=0)


print(Xmat.columns.values)
print(str(Xmat.shape[1]) + ' columns')

# add a new feature 'total sqfootage'
Xmat['TotalSF'] = Xmat['TotalBsmtSF'] + Xmat['1stFlrSF'] + Xmat['2ndFlrSF']
print('There are currently ' + str(Xmat.shape[1]) + ' columns.')

y_train = np.log(y_train)

# train and test
X_train = Xmat.iloc[:train.shape[0],:]
X_test = Xmat.iloc[train.shape[0]:,:]

# Compute the correlation matrix
corr = X_train.corr()

# feature importance using random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X_train, y_train)
print('Training done using Random Forest')
ranking = np.argsort(-rf.feature_importances_)
# use the top 30 features only
X_train = X_train.iloc[:,ranking[:30]]
X_test = X_test.iloc[:,ranking[:30]]

# interaction between the top 2
X_train["Interaction"] = X_train["TotalSF"]*X_train["OverallQual"]
X_test["Interaction"] = X_test["TotalSF"]*X_test["OverallQual"]

# zscoring
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()

# outlier deletion
Xmat = X_train
Xmat['SalePrice'] = y_train
Xmat = Xmat.drop(Xmat[(Xmat['TotalSF']>5) & (Xmat['SalePrice']<12.5)].index)
Xmat = Xmat.drop(Xmat[(Xmat['GrLivArea']>5) & (Xmat['SalePrice']<13)].index)

# recover
y_train = Xmat['SalePrice']
X_train = Xmat.drop(['SalePrice'], axis=1)
train_dataset = pd.concat([X_train, y_train], axis=1)

file_name = 'housepnr2.xlsx'
# saving the excel
train_dataset.to_excel(file_name)
print('DataFrame is written to Excel File successfully.')
aaaa=1

























# print(train.head())
# train_id = train['Id']
# test_id = test['Id']
# del train['Id']
# del test['Id']
#
# train1 = train.copy()
# train1 = train1.drop(train1[(train1['GarageArea']>1200) & (train1['SalePrice']<300000)].index)
# train1 = train1.drop(train1[(train1['GrLivArea']>4000) & (train1['SalePrice']<300000)].index)
# train1 = train1.drop(train1[(train1['TotalBsmtSF']>5000)].index)
#
# X = train1.drop('SalePrice', axis=1)
# y = train1['SalePrice'].to_frame()
#
# # Add variable
# X['train'] = 1
# test['train'] = 0
#
# # Combining train and test for data cleaning
# df = pd.concat([test, X])
# aaa=1
# print('Count of Features per Data Type:')
# print(df.dtypes.value_counts())
#
# # Do we have duplicates?
# print('Number of Duplicates:', len(df[df.duplicated()]))
#
# # Do we have missing values?
# print('Number of Missing Values:', df.isnull().sum().sum())
#
# print('Missing Values per Column:')
# print(df.isnull().sum().sort_values(ascending=False).head(25))
# df['PoolQC'] = df['PoolQC'].fillna('None')
# df['MiscFeature'] = df['MiscFeature'].fillna('None')
# df['Alley'] = df['Alley'].fillna('None')
# df['Fence'] = df['Fence'].fillna('None')
# df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
# df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda i: i.fillna(i.median()))
# # Let's take a look at the "Garage" features
# garage_cols = [col for col in df if col.startswith('Garage')]
# print(df[garage_cols])
#
# # For the numerical features:
# for i in df[garage_cols].select_dtypes(exclude='object').columns:
#     df[i] = df[i].fillna(0)
#
# # For the categorical features:
# for i in df[garage_cols].select_dtypes(include='object').columns:
#     df[i] = df[i].fillna('None')
#
#
# bsmt_cols = [col for col in df if col.startswith('Bsmt')]
#
# # For the numerical features:
# for i in df[bsmt_cols].select_dtypes(exclude='object').columns:
#     df[i] = df[i].fillna(0)
#
# # For the categorical features:
# for i in df[bsmt_cols].select_dtypes(include='object').columns:
#     df[i] = df[i].fillna('None')
#
# mas_cols = [col for col in df if col.startswith('Mas')]
#
# # For the numerical features:
# for i in df[mas_cols].select_dtypes(exclude='object').columns:
#     df[i] = df[i].fillna(0)
#
# # For the categorical features:
# for i in df[mas_cols].select_dtypes(include='object').columns:
#     df[i] = df[i].fillna('None')
#
# df['MSZoning'] = df.groupby('Neighborhood')['MSZoning'].transform(lambda i: i.fillna(i.value_counts().index[0]))
#
# print('Missing Values left:')
# print(df.isnull().sum().sort_values(ascending=False).head(10))
#
# # replace missing values for mode of each column
# df = df.fillna(df.mode().iloc[0])
#
# print(df.describe().T)
#
# df['MSSubClass'] = df['MSSubClass'].astype(str)
# df['MoSold'] = df['MoSold'].astype(str)           # months is always categorical
# df['YrSold'] = df['YrSold'].astype(str)           # year sold just have 5 years
#
# df['Total_House_SF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
# df['Total_Home_Quality'] = (df['OverallQual'] + df['OverallCond'])/2
# df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
#
# numeric_cols = df.select_dtypes(exclude='object').columns
#
# skew_limit = 0.5
# skew_vals = df[numeric_cols].skew()
#
# skew_cols = (skew_vals
#              .sort_values(ascending=False)
#              .to_frame()
#              .rename(columns={0:'Skew'})
#              .query('abs(Skew) > {0}'.format(skew_limit)))
#
# print(skew_cols)
# from scipy.special import boxcox1p
# from scipy.stats import boxcox_normmax
#
# # Normalize skewed features
# for col in skew_cols.index:
#     df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))
#
# # log(1+x) transform
# y["SalePrice"] = np.log1p(y["SalePrice"])
#
# categ_cols = df.dtypes[df.dtypes == np.object]        # filtering by categorical variables
# categ_cols = categ_cols.index.tolist()                # list of categorical fields
#
# df_enc = pd.get_dummies(df, columns=categ_cols, drop_first=True)   # One hot encoding
#
# X = df_enc[df_enc['train']==1]
# test = df_enc[df_enc['train']==0]
# X.drop(['train'], axis=1, inplace=True)
# test.drop(['train'], axis=1, inplace=True)
#
#
#
#
# file_name = 'housepnr.xlsx'
# # saving the excel
# df_enc.to_excel(file_name)
# print('DataFrame is written to Excel File successfully.')
#
#
# X = df_enc[df_enc['train']==1]
# test = df_enc[df_enc['train']==0]
# X.drop(['train'], axis=1, inplace=True)
# test.drop(['train'], axis=1, inplace=True)
#
# from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)