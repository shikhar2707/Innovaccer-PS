#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### Importing our Dataset

# In[2]:


CancerData = pd.read_excel("Innovaccer/BreastCancer_Prognostic_v1.xlsx")
CancerData


# In[3]:


columns_list = list(CancerData.columns)
columns_list.remove("ID")
new = columns_list[2:] + columns_list[:2]
new
CancerData = CancerData[new]


# In[33]:


CancerData = CancerData[CancerData["Lymph_Node_Status"] != "?"]
CancerData["Lymph_Node_Status"] = CancerData["Lymph_Node_Status"].astype("int64")
CancerData.Outcome[CancerData["Outcome"] == "R"] = 1
CancerData.Outcome[CancerData["Outcome"] == "N"] = 0
CancerData.Outcome = CancerData.Outcome.astype("int64")


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr_data = CancerData.corr()
plt.subplots(figsize = (20,15))
sns.heatmap(corr_data, xticklabels=corr_data.columns, yticklabels=corr_data.columns, annot=True, fmt = ".1g",vmin = -1,vmax = 1,center = 0,cmap = "coolwarm")
plt.savefig("Heatmap.jpg")


# ### Important points to note - 
# #### 1. Since Area and Perimeter are a function of radius. hence they have high correlation with the radius.
# #### 2. Our cutoff Correlation coefficient range must be -0.8 to 0.8

# In[7]:


from sklearn.metrics import confusion_matrix, f1_score


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr_data = CancerData_1.corr()
plt.subplots(figsize = (20,15))
sns.heatmap(corr_data, xticklabels=corr_data.columns, yticklabels=corr_data.columns, annot=True, fmt = ".1g",vmin = -1,vmax = 1,center = 0,cmap = "coolwarm")
plt.savefig("Used features Heatmap.jpg")


# ## Features

# In[36]:


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


# ## Our Classifier

# In[37]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train)


# In[38]:


y_pred = clf_gini.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_gini.score(X_test, y_test)))


# ## Confusion Matrix

# In[39]:


cm = confusion_matrix(y_test,y_pred)
cm


# ## Recall Score

# In[42]:


from sklearn.metrics import precision_recall_fscore_support
score = precision_recall_fscore_support(y_test, y_pred,average = "weighted")
print(score[1])


# ### Saving the predictions alongside the original values to a csv file

# In[43]:


y_test


# In[45]:


y_pred  = list(y_pred)


# In[48]:


Classified = pd.DataFrame()
Classified["Actual"] = y_test
Classified["Predicted"] = y_pred
Classified.to_csv("Innovaccer/Classification/Classified.csv")


# In[ ]:




