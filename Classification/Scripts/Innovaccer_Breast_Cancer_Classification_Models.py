#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np


# ### Reading our data file which is in .xlsx i.e Excel Sheet format

# In[80]:


CancerData = pd.read_excel("Innovaccer/BreastCancer_Prognostic_v1.xlsx")
CancerData


# In[81]:


#There are 35 features extracted from the tumors.
CancerData.columns.size


# In[82]:


#List of features/columns
CancerData.columns


# # Dataset Preparation

# #### Apparently there is not a single null value in our dataset.

# In[83]:


CancerData.isnull().sum()


# In[84]:


# For our ease lets chamge the order of our dataframe and drop that ID frame
columns_list = list(CancerData.columns)
columns_list.remove("ID")
new = columns_list[2:] + columns_list[:2]
new
CancerData = CancerData[new]


# In[85]:


CancerData


# In[86]:


CancerData.dtypes


# In[87]:


# Since that '?' string will cause troubles in our analysis
CancerData = CancerData[CancerData["Lymph_Node_Status"] != "?"]
CancerData["Lymph_Node_Status"] = CancerData["Lymph_Node_Status"].astype("int64")
CancerData.Outcome[CancerData["Outcome"] == "R"] = 1
CancerData.Outcome[CancerData["Outcome"] == "N"] = 0
CancerData.Outcome = CancerData.Outcome.astype("int64")


# In[88]:


#Lets see the statistical data of each feature
CancerData.describe()


# ## Analyzing the data using correlation plot

# In[89]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr_data = CancerData.corr()
plt.subplots(figsize = (20,15))
sns.heatmap(corr_data, xticklabels=corr_data.columns, yticklabels=corr_data.columns, annot=True, fmt = ".1g",vmin = -1,vmax = 1,center = 0,cmap = "coolwarm")


# ### Important points to note - 
# #### 1. Since Area and Perimeter are a function of radius. hence they have high correlation with the radius.
# #### 2. Our cutoff Correlation coefficient range must be -0.8 to 0.8

# # Classification without feature selection

# In[90]:


X = CancerData[['Time', 'radius_mean', 'texture_mean',
       'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
       'concavity_mean', 'concave_points_mean', 'symmetry_mean',
       'fractal_dimension_mean', 'radius_std_dev', 'texture_std_dev',
       'perimeter_std_dev', 'area_std_dev', 'smoothness_std_dev',
       'compactness_std_dev', 'concavity_std_dev', 'concave_points_std_dev',
       'symmetry_std_dev', 'fractal_dimension_std_dev', 'Worst_radius',
       'Worst_texture', 'Worst_perimeter', 'Worst_area', 'Worst_smoothness',
       'Worst_compactness', 'Worst_concavity', 'Worst_concave_points',
       'Worst_symmetry', 'Worst_fractal_dimension', 'Tumor_Size',
       'Lymph_Node_Status']]
Y = CancerData["Outcome"]


# In[91]:


from sklearn.metrics import confusion_matrix, f1_score


# ### 1. Logistic Regression

# In[92]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
logreg = LogisticRegression(max_iter = 100)
logreg.fit(X_train, y_train)


# In[93]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[94]:


cm = confusion_matrix(y_test,y_pred)
cm


# ### 2. Decision Tree

# In[95]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train)


# In[96]:


y_pred = clf_gini.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))


# In[97]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[98]:


from sklearn.metrics import precision_recall_fscore_support
score = precision_recall_fscore_support(y_test, y_pred,average = "weighted")
print(score)


# ### 3. Support Vector Machine

# In[99]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
clf.fit(X_train, y_train)


# In[100]:


y_pred = clf.predict(X_test)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# In[101]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[102]:


from sklearn.metrics import precision_recall_fscore_support
score = precision_recall_fscore_support(y_test, y_pred,average = "weighted")
print(score)


# ### 4. Random Forest
# 

# In[103]:


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state = 0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)


# In[104]:


y_pred=clf.predict(X_test)


# In[105]:


print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# In[106]:


from sklearn.metrics import precision_recall_fscore_support
score = precision_recall_fscore_support(y_test, y_pred,average = "weighted")
print(score)


# ### 5 . Naive Bayes

# In[107]:


from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)


# In[108]:


clf = GaussianNB()
clf.fit(X_train,y_train)


# In[109]:


print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# # Feature Selection 1

# In[120]:


# Dropping the features which are out of the acceptable corr. coeff. range i.e (-0.8 , 0.8)


# In[121]:


CancerData_1 = CancerData.drop(["perimeter_mean" , "area_mean" , "Worst_radius","Worst_perimeter","Worst_area","Worst_texture","concavity_mean","Worst_compactness","Worst_concave_points","Worst_fractal_dimension","perimeter_std_dev","area_std_dev","fractal_dimension_std_dev","concave_points_std_dev","concavity_std_dev"],axis = 1)


# In[122]:


#New Plot looks more varied now
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr_data = CancerData_1.corr()
plt.subplots(figsize = (20,15))
sns.heatmap(corr_data, xticklabels=corr_data.columns, yticklabels=corr_data.columns, annot=True, fmt = ".1g",vmin = -1,vmax = 1,center = 0,cmap = "coolwarm")


# In[123]:


CancerData_1.columns


# ## Let us build our classifier model now

# ### 1. Logistic regression

# In[124]:


X = CancerData_1[['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_std_dev', 'texture_std_dev', 'smoothness_std_dev',
       'compactness_std_dev', 'symmetry_std_dev', 'Worst_smoothness',
       'Worst_concavity', 'Worst_symmetry', 'Tumor_Size', 'Lymph_Node_Status','Time']]
Y = CancerData_1["Outcome"]


# In[125]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[126]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[127]:


cm = confusion_matrix(y_test,y_pred)
cm


# ### 2. Decision Trees

# In[128]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train)


# In[129]:


y_pred = clf_gini.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))


# In[130]:


cm = confusion_matrix(y_test,y_pred)
cm


# ### 3. Support Vector Machine

# In[131]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
clf.fit(X_train, y_train)


# In[132]:


y_pred = clf.predict(X_test)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# In[133]:


cm = confusion_matrix(y_test,y_pred)
cm


# ### 4. Random Forest

# In[134]:


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state = 0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)


# In[135]:


print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# ### 5. Naive Bayes

# In[136]:


from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


# In[137]:


clf = GaussianNB()
clf.fit(X_train,y_train)


# In[138]:


print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# # Feature Selection 2

# ### ANOVA Statistics

# In[485]:


X = CancerData[['Time', 'radius_mean', 'texture_mean',
       'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
       'concavity_mean', 'concave_points_mean', 'symmetry_mean',
       'fractal_dimension_mean', 'radius_std_dev', 'texture_std_dev',
       'perimeter_std_dev', 'area_std_dev', 'smoothness_std_dev',
       'compactness_std_dev', 'concavity_std_dev', 'concave_points_std_dev',
       'symmetry_std_dev', 'fractal_dimension_std_dev', 'Worst_radius',
       'Worst_texture', 'Worst_perimeter', 'Worst_area', 'Worst_smoothness',
       'Worst_compactness', 'Worst_concavity', 'Worst_concave_points',
       'Worst_symmetry', 'Worst_fractal_dimension', 'Tumor_Size',
       'Lymph_Node_Status']]
Y = CancerData["Outcome"]


# In[486]:


from sklearn.feature_selection import f_classif
var_rel = f_classif(X,Y)
var_rel


# In[487]:


from sklearn.feature_selection import SelectKBest
K_Best_Features = SelectKBest(f_classif , k = 10)
features_new = K_Best_Features.fit_transform(X,Y)
features_new.shape
X_new = features_new


# ## Lets build our classifier model now

# ### 1. Logistic Regression

# In[432]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[433]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[434]:


cm = confusion_matrix(y_test,y_pred)
cm


# ### 2. Decision Trees

# In[435]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size = 0.3, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train)


# In[436]:


y_pred = clf_gini.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))


# In[437]:


cm = confusion_matrix(y_test,y_pred)
cm


# ### 3. Support Vector Machine

# In[438]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.25, random_state=0)
clf.fit(X_train, y_train)


# In[439]:


y_pred = clf.predict(X_test)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# In[440]:


cm = confusion_matrix(y_test,y_pred)
cm


# ## Final Classification Model:
# ### Decision tree Algorithm with all the features used.

# In[ ]:




