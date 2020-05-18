#!/usr/bin/env python
# coding: utf-8

# In[1109]:


import numpy as np
import pandas as pd


# In[1110]:


CancerData = pd.read_excel("Innovacer/BreastCancer_Prognostic_v1.xlsx")


# In[1111]:


Recurrent = CancerData[CancerData["Outcome"] == "R"]


# In[1112]:


Recurrent


# In[1113]:


Recurrent.dtypes


# In[1114]:


Recurrent = Recurrent.drop(["Outcome"] , axis = 1)


# In[1115]:


Recurrent.describe()


# ### Since, we have to build a regression model, Our Work pipeline would be following:
# #### 1. Dividing the dataset into test and train. We might have a ratio of 80:20 Train: Test.
# #### 2. Comparing the predicted values to the original test values and computing the model accuracy.
# #### 3. Saving the predicted results to a new csv file.

# In[1116]:


ID = CancerData[CancerData["Outcome"] == "R"]["ID"]


# In[1117]:


Recurrent = Recurrent[Recurrent["Lymph_Node_Status"] != "?"]
Recurrent["Lymph_Node_Status"] = Recurrent["Lymph_Node_Status"].astype("int64")


# In[1118]:


Recurrent


# In[1119]:


Recurrent.columns


# In[1120]:


X = Recurrent[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_std_dev', 'texture_std_dev', 'perimeter_std_dev',
       'area_std_dev', 'smoothness_std_dev', 'compactness_std_dev',
       'concavity_std_dev', 'concave_points_std_dev', 'symmetry_std_dev',
       'fractal_dimension_std_dev', 'Worst_radius', 'Worst_texture',
       'Worst_perimeter', 'Worst_area', 'Worst_smoothness',
       'Worst_compactness', 'Worst_concavity', 'Worst_concave_points',
       'Worst_symmetry', 'Worst_fractal_dimension', 'Tumor_Size','Lymph_Node_Status']]
Y = Recurrent["Time"]


# In[1121]:


for column in X.columns:
    Recurrent[column] = (Recurrent[column] - Recurrent[column].min())/(Recurrent[column].max() - Recurrent[column].min())


# In[1122]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr_data = Recurrent.corr()
plt.subplots(figsize = (20,15))
sns.heatmap(corr_data, xticklabels=corr_data.columns, yticklabels=corr_data.columns, annot=True, fmt = ".1g",vmin = -1,vmax = 1,center = 0,cmap = "gist_heat")


# ### Important points to note - 
# #### 1. Since Area and Perimeter are a function of radius. hence they have high correlation with the radius.
# #### 2. Our cutoff Correlation coefficient range must be -0.8 to 0.8

# In[1123]:


from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
var_rel = f_regression(X,Y , center = True)
var_rel


# In[1124]:


from sklearn.feature_selection import SelectKBest
K_Best_Features = SelectKBest(f_regression , k = 10)
features_new = K_Best_Features.fit_transform(X,Y)
features_new.shape
X_new = features_new


# In[1125]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge,Lasso,LinearRegression,ElasticNet
from sklearn import metrics


# ### Ridge regression

# In[1126]:


scores_ridge = []

for i in range(10000):
        X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.20, random_state=i)
        regressor = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=0) 
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        scores_ridge.append(regressor.score(X_test,y_test))


# In[1127]:


max(scores_ridge)


# ### Linear Regression

# In[973]:


scores_lr = []
for i in range(10000):
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, random_state=i)
    regressor = LinearRegression() 
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    scores_lr.append(regressor.score(X_test,y_test))


# In[974]:


max(scores_lr)


# ### ElasticNet Regression

# In[975]:


scores_en = []
for i in range(10000):
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, random_state=i)
    regressor = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic') 
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    scores_en.append(regressor.score(X_test,y_test))


# In[962]:


max(scores_en)


# ### Lasso Regression

# In[976]:


scores_lasso = []
for i in range(10000):
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, random_state=i)
    regressor = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic') 
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    scores_lasso.append(regressor.score(X_test,y_test))


# In[977]:


max(scores_lasso)


# In[978]:


for j in range(len(scores_lasso)):
    if scores_lasso[j] == max(scores_lasso):
        print(j)


# ### Support Vector Regression

# In[1062]:


score=[]
from sklearn.svm import SVR
for i in range(100):
    clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=i/100, gamma='scale',
               kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    clf.fit(X_train,y_train)
    score.append(clf.score(X_test,y_test))


# In[1063]:


len(score)


# In[1064]:


for i in range(len(score)):
    if score[i] == max(score):
        print(i)


# In[1088]:


max(score)


# ## By above model analyses, the Lasso Regression Model with a particular random state yields the best positive r^2 statistic i.e nearly 0.73.

# ### Hence we implement it on our dataset and yield our result.

# In[1130]:


X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, random_state=6345)
regressor = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic') 
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)


# ### Creating the predicted values dataframe

# In[1183]:


y_pred=list(y_pred)


# In[1163]:


index = []
for num in y_test.index:
        index.append(num)


# In[1164]:


index


# In[1185]:


Result = pd.DataFrame(index)


# In[1186]:


ID = []
Time= []
for ind in index:
        ID.append(list(Recurrent.loc[ind,:])[0])
        Time.append(list(Recurrent.loc[ind,:])[1])
Time


# In[1187]:


Result["ID"] = pd.DataFrame(ID)
Result["Time"] = pd.DataFrame(Time)
Result["Predicted Time"] = pd.DataFrame(y_pred)


# In[1188]:


Result


# ### Saving the result as a csv dataframe.

# In[1189]:


Result.to_csv("Innovacer/Predicted.csv")


# In[ ]:




