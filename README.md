# A3_Elnaz_Askari_GAMLAB
This package is written as the third project for GAMLAB artificial intelligence course.
import numpy as np
from sklearn.datasets import fetch_california_housing
data=fetch_california_housing()


#step0-----> data ha clean hastan

#step1-----> tafkik x , y 
x=data.data
y=data.target

data.feature_names
data.target_names

'''
x--->
    ['MedInc',
     'HouseAge',
     'AveRooms',
     'AveBedrms',
     'Population',
     'AveOccup',
     'Latitude',
     'Longitude']

y--->gheymate khone
    
20640 khone
'''


#==============================================================================
#===============================LinearRegression===============================
#==============================================================================


#step2-----> kfold
from sklearn.model_selection import KFold
kf=KFold(n_splits=25,shuffle=True,random_state=42)

#step3-----> model = LinearRegression
from sklearn.linear_model import LinearRegression
model=LinearRegression() #hyperparameter nadarad
my_params={}

#step4-----> GridSeachCV
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)

#step5-----> best_score and predict
gs.best_score_ 
#-0.31747504835454854

#predict
gs.predict(np.array([5,20,5,1,1000,2,25,-122]).reshape(1,-1))
#8.00498646


#==============================================================================
#=============================KNeighborsRegressor==============================
#==============================================================================


#step2-----> kfold
from sklearn.model_selection import KFold
kf=KFold(n_splits=25,shuffle=True,random_state=42)

#step3-----> model = KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
model=KNeighborsRegressor()
my_params= { 'n_neighbors':[1,2,3,4,5,9,10,15,25,50,75,100],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }

#step4-----> GridSeachCV
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)

#step5-----> best_score and best_params and predict
gs.best_score_ 
#-0.4782839611791587
gs.best_params_
#{'metric': 'manhattan', 'n_neighbors': 4}

#predict
gs.predict(np.array([5,20,5,1,1000,2,25,-122]).reshape(1,-1))
#2.077

#==============================================================================
#=============================DecisionTreeRegressor============================
#==============================================================================

#step2-----> kfold
from sklearn.model_selection import KFold
kf=KFold(n_splits=25,shuffle=True,random_state=42)

#step3-----> model = DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(random_state=42)
my_params= {'max_depth':[1,3,5,7,10,15,16,17,18,19,20,25,50]}


#step4-----> GridSeachCV
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)

#step5-----> best_score and best_params and predict
gs.best_score_ 
#-0.23778052777642497
gs.best_params_
#{'max_depth': 15}

#predict
gs.predict(np.array([5,20,5,1,1000,2,25,-122]).reshape(1,-1))
#4.81700333
'''
*****min_samples_split*****----->
----> hadeaghal tedade nemone haee k naiz has ta taghsim kone tu har daste

ye ghesmati az total sample (20  --> 10 , 
                            45 --> 15 ya 3 ya 5)

*****min_samples_leaf*****----->
-----> taghsim mikone har soal _.chanta soal ? (1,2,5)
'''

#==============================================================================
#=============================RandomForestRegressor============================
#==============================================================================

#step2-----> kfold
from sklearn.model_selection import KFold
kf=KFold(n_splits=25,shuffle=True,random_state=42)


#step3-----> model = RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=42)
my_params= {'max_depth':[15,20,25,35,45,50,55,65,75],
            'max_features':[1,2,3,4,5,6,7,10,15],
            'n_estimators':[100,120,150]}


#step4-----> GridSeachCV
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error',n_jobs=-1)

gs.fit(x,y)

#step5-----> best_score and best_params and predict
gs.best_score_ 
#-0.17727691857081904
gs.best_params_
#{'max_depth': 45, 'max_features': 4, 'n_estimators': 150}

#predict
gs.predict(np.array([5,20,5,1,1000,2,25,-122]).reshape(1,-1))
#3.55450833

#==============================================================================
#=============================SVR==============================================
#==============================================================================

#step2-----> kfold
from sklearn.model_selection import KFold
kf=KFold(n_splits=5,shuffle=True,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
x_scaled = scaler.fit_transform(x)

#step3-----> model = SVR
from sklearn.svm import SVR
model=SVR()
my_params={'kernel':['poly','rbf'],
           'C':[50,70,100],
           'degree':[2,3,4]}


#step4-----> GridSeachCV
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error',n_jobs=-1)

gs.fit(x_scaled,y)

#step5-----> best_score and best_params and predict
gs.best_score_
#-0.21320734925094165
gs.best_params_
#{'C': 100, 'degree': 2, 'kernel': 'rbf'}

#predict
gs.predict(np.array([5,20,5,1,1000,2,25,-122]).reshape(1,-1))
#6.19796439


#va ghyeymat ro bar asase yek kodom az moalefe ha rasm mikoni (pishbinish ro)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
data=fetch_california_housing()


x=np.array(data.data[:,0]).reshape(-1,1) 
y=np.array(data.target[:])




from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.25,shuffle=True,random_state=42) 

#rasm
plt.scatter(x_test,y_test,label='test dataset')
plt.scatter(x_train,y_train,label='train dataset')
plt.title('DATA')
plt.xlabel('MedInc')
plt.ylabel('cost')
plt.legend()
plt.grid()
plt.show()

##############
#step2-----> kfold
from sklearn.model_selection import KFold
kf=KFold(n_splits=25,shuffle=True,random_state=42)

#step3-----> model = DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(random_state=42)
my_params= {'max_depth':[15]}


#step4-----> GridSeachCV
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)
###############

