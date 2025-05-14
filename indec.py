import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
import joblib


data = pd.read_csv('USA Housing Dataset.csv')

data = data.drop(columns=['street', 'city', 'statezip', 'country','view','condition','yr_built','yr_renovated','date'])
y = np.log1p(data['price'])
print(data.head())
X = data.drop(columns=['price'])
plt.figure(figsize=(15, 10))
for i,fitur in enumerate(X.columns):
    plt.subplot(4, 3, i+1)
    plt.scatter(X[fitur],y)
    plt.title(f'price vs {fitur}')
plt.tight_layout()
plt.show()
z_scores = np.abs(zscore(X.select_dtypes(include=["int64", "float64"])))
z_outliers = (z_scores > 3).any(axis=1)
# print(z_outliers.sum())
# Remove outliers
X = X[~ z_outliers]
y = y[~z_outliers]

# Normalize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
x_scaled = pd.DataFrame(x_scaled,columns=X.columns,index=X.index)
print(x_scaled.head())
#linear regression
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)
# linear = LinearRegression()
# linear.fit(x_train,y_train)
# y_pred = linear.predict(x_test)



# Random Forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
random_forest.fit(x_train,y_train)
y_preds = random_forest.predict(x_test)

#mse
# y_pred = np.expm1(y_pred)
y_preds = np.expm1(y_preds)
y_test = np.expm1(y_test)
# mse = mean_squared_error(y_test,y_pred)
# rmse = np.sqrt(mse)
msee = mean_squared_error(y_test,y_preds)
rmsee = np.sqrt(msee)

# #cross validation
# linear_cross_val_scores = cross_val_score(linear,x_scaled,y,cv = 6 ,scoring='neg_mean_squared_error')
random_forestcross = cross_val_score(random_forest,x_scaled,y,cv = 6 ,scoring='neg_mean_squared_error')
# linear_cross_val_scores  = np.sqrt(-linear_cross_val_scores)
random_forestcross = np.sqrt(-random_forestcross)
# print(f'linear regression cross validation mean squared error: {linear_cross_val_scores.mean()} +- {linear_cross_val_scores.std()}')
print(f'Random forest cross validation mean squared error: {random_forestcross.mean()} +- {random_forestcross.std()}')
print(f'Random Forest Mean Squared Error: {msee}')
print(f'Random Forest Root Mean Squared Error: {rmsee}')

data_baru = [[3,2,1340,1384,3,0,1340,0]]
data_baru = pd.DataFrame(data_baru,columns=x_train.columns)
data_baru = scaler.transform(data_baru)
data_baru = pd.DataFrame(data_baru,columns=x_train.columns)
print(data_baru.head())

data_baru = random_forest.predict(data_baru)
data_baru = np.expm1(data_baru)
print(data_baru)

joblib.dump(random_forest, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Menyimpan nama kolom fitur yang digunakan dalam pelatihan model
joblib.dump(x_train.columns, 'features.pkl')
