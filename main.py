# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Reading Csv
data = pd.read_csv("maaslar_yeni.csv")
print(data)

x = data.iloc[:,2:5]
y = data.iloc[:,5:]

X = x.values
Y = y.values
    
print(data.corr())

predictceo = [[10, 10, 100]]
predictceodf = pd.DataFrame(predictceo, columns = ["TitleLevel","Seniority","Point"])
predictmanager = [[7, 10, 100]]
predictmanagerdf = pd.DataFrame(predictmanager, columns = ["TitleLevel","Seniority","Point"])

# Gerekli değişkenleri p value ile bakıyoruz
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

print("Linear OLS")
model = sm.OLS(lin_reg.predict(x), x)
print(model.fit().summary())

#--------------------------------------------------------------------------------
# Multiple Linear Regression

# Eğitim verisi ve Test verisi olarak ayırma
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

L_predceo = lr.predict(predictceodf)
L_predmanager = lr.predict(predictmanagerdf)

#--------------------------------------------------------------------------------
# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(X)
lpoly_reg = LinearRegression()
lpoly_reg.fit(x_poly, y)

print("Polynomial OLS")
modelpoly = sm.OLS(lpoly_reg.predict(x_poly), x)
print(modelpoly.fit().summary())

P_predceo = lpoly_reg.predict(poly_reg.fit_transform(predictceodf))
P_predmanager = lpoly_reg.predict(poly_reg.fit_transform(predictmanagerdf))
#--------------------------------------------------------------------------------
# Support Vector Regression
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_scaled = sc.fit_transform(X)

sc2=StandardScaler()
Y_scaled = sc2.fit_transform(Y)

from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(X_scaled, Y_scaled)

print("SVR OLS")
modelsvr = sm.OLS(svr_reg.predict(X_scaled), X_scaled)
print(modelsvr.fit().summary())

svr_predceo = svr_reg.predict(sc.fit_transform(predictceodf))
svr_predmanager = svr_reg.predict(sc.fit_transform(predictmanagerdf))
#--------------------------------------------------------------------------------
# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_r = DecisionTreeRegressor(random_state = 0)
dt_r.fit(x, y)

print("DT OLS")
modeldt = sm.OLS(dt_r.predict(x), x)
print(modeldt.fit().summary())

dt_predceo = dt_r.predict(predictceodf)
dt_predmanager = dt_r.predict(predictmanagerdf)
#--------------------------------------------------------------------------------
# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=30, random_state = 0)
rf_reg.fit(X, Y.ravel())

print("RF OLS")
modelrf = sm.OLS(rf_reg.predict(X), X)
print(modelrf.fit().summary())

rf_predceo = rf_reg.predict(predictceodf)
rf_predmanager = rf_reg.predict(predictmanagerdf)















