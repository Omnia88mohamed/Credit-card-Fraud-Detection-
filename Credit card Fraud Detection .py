#!/usr/bin/env python
# coding: utf-8

# #                                     Credit card Fraud Detection

# ## Importing Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, datasets
from sklearn.tree import DecisionTreeClassifier
import pickle
import joblib


# ## Importing Dataset

# In[2]:


credit_card_data = pd.read_csv("D:./4th/creditcard.csv")


# # Data Understanding & Preprocessing

# ## first 5 rows of the dataset

# In[3]:


credit_card_data.head()


# ## last 5 rows of the dataset

# In[4]:


credit_card_data.tail()


# ## dataset informations

# In[5]:


credit_card_data.info()


# ## checking the number of missing values in each column

# In[6]:


credit_card_data.isnull().sum()


# ###### there is no null values

# ## Checking Distribution of legit transactions & fraudulent transactions

# In[7]:


credit_card_data['Class'].value_counts()


# 0 Means --> Normal Transaction
# 
# 1 Means --> fraudulent transaction
# ###### That's  mean that This Dataset is highly unblanced

# ## separating the data for analysis

# In[8]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# ## statistical measures of the data

# In[10]:


legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# ## compare the values for both transactions

# In[12]:


credit_card_data.groupby('Class').mean()


# In[13]:


sns.countplot(x='Class', data=credit_card_data)


# ### Under-Sampling
# Building a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# 
# Number of Fraudulent Transactions --> 492
# 
# 

# In[14]:


legit_sample = legit.sample(n=492)


# ## Concatenating two DataFrames

# In[15]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[16]:


new_dataset.head()


# In[17]:


new_dataset['Class'].value_counts()


# In[18]:


new_dataset.groupby('Class').mean()


# ## Splitting the data into Features & Targets

# In[19]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)


# In[20]:


print(Y)


# ## Split the data into Training data & Testing Data

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=42)


# In[22]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training

# ## 1-Logistic Regression

# In[23]:


logistic_model = LogisticRegression()


# ### training the Logistic Regression Model with Training Data

# In[24]:


logistic_model.fit(X_train, Y_train)


# ### Model Evaluation
# 

# #### Accuracy Score

# ##### accuracy on training data

# In[25]:


Xl_train_prediction = logistic_model.predict(X_train)
training_data_accuracy = accuracy_score(Xl_train_prediction, Y_train)


# In[26]:


print('Accuracy on Training data  of logistic  model : ', training_data_accuracy)


# ##### accuracy on test data

# In[27]:


Xl_test_prediction = logistic_model.predict(X_test)
lac = accuracy_score(Xl_test_prediction, Y_test)


# In[28]:


print('Accuracy score on Test Data of logistic model: ', lac)


# ## 2-Decision Tree Classifier

# In[29]:


DT = DecisionTreeClassifier( max_depth=5)


# ### training the Decision Tree Classifier  Model with Training Data

# In[30]:


DT.fit(X_train, Y_train)


# ### Model Evaluation

# #### Accuracy Score

# ##### accuracy on training data¶

# In[31]:


Xd_train_prediction = DT.predict(X_train)
trainingd_data_accuracy = accuracy_score(Xd_train_prediction, Y_train)


# In[32]:


print('Accuracy on Training data  of Decision Tree  model : ', trainingd_data_accuracy)


# #### accuracy on test data

# In[33]:


Xd_test_prediction = DT.predict(X_test)
dtac= accuracy_score(Xd_test_prediction, Y_test)


# In[34]:


print('Accuracy score on Test Data of Decision Tree model: ', dtac)


# # 3-K-Nearest Neighbors

# In[35]:


knn = KNeighborsClassifier(n_neighbors=5)


# ### training the K-Nearest Neighbors Model with Training Data

# In[36]:


knn.fit(X_train, Y_train)


# ### Model Evaluation

# #### Accuracy Score

# #### accuracy on training data

# In[37]:


Xk_train_prediction = knn.predict(X_train)
trainingk_data_accuracy = accuracy_score(Xk_train_prediction, Y_train)


# In[38]:


print('Accuracy on Training data  of K-Nearest Neighbors model : ', trainingk_data_accuracy)


# #### accuracy on test data

# In[39]:


Xk_test_prediction = knn.predict(X_test)
kac = accuracy_score(Xk_test_prediction, Y_test)


# In[40]:


print('Accuracy score on Test Data of K-Nearest Neighbors model: ', kac)


# # 4-Random Forest Classifier  model

# In[41]:


Rclf = RandomForestClassifier(max_depth=7)


# ### training the Random Forest Classifier Model with Training Data

# In[42]:


Rclf.fit(X_train, Y_train)


# ### Model Evaluation¶

# ### Accuracy Score

# #### accuracy on training data

# In[43]:


Xr_train_prediction = Rclf.predict(X_train)
trainingr_data_accuracy = accuracy_score(Xr_train_prediction, Y_train)


# In[44]:


print('Accuracy on Training data  of Random Forest Classifier  model : ', trainingr_data_accuracy)


# #### accuracy on test data
# 

# In[45]:


Xr_test_prediction = Rclf.predict(X_test)
rac = accuracy_score(Xr_test_prediction, Y_test)


# In[46]:


print('Accuracy score on Test Data of Random Forest Classifier model: ', rac)


# # 5-Gaussian Naive Bayes

# In[47]:


gs=GaussianNB()


# In[48]:


gs.fit(X_train, Y_train)


# ### training the Gaussian Naive Bayes  Model with Training Data

# In[49]:


Xg_train_prediction = gs.predict(X_train)
trainingg_data_accuracy = accuracy_score(Xg_train_prediction, Y_train)


# ### Model Evaluation¶

# ### Accuracy Score

# #### accuracy on training data

# In[50]:


print('Accuracy on Training data  of Gaussian Naive Bayes  model : ', trainingg_data_accuracy)


# #### accuracy on test data

# In[51]:


Xg_test_prediction = gs.predict(X_test)
gac = accuracy_score(Xg_test_prediction, Y_test)


# In[52]:


print('Accuracy score on Test Data of Gaussian Naive Bayes model: ', gac)


# ## 6- support vector machines

# In[53]:


Svm = svm.SVC()


# ### training the support vector machines Model with Training Data

# In[54]:


Svm.fit(X_train, Y_train)


# ### Model Evaluation

# ### Accuracy Score

# ### accuracy on training data

# In[55]:


Xs_train_prediction = Svm.predict(X_train)
trainings_data_accuracy = accuracy_score(Xs_train_prediction, Y_train)


# In[56]:


print('Accuracy on Training data  of support vector machines  model : ', trainings_data_accuracy)


# ### accuracy on test data

# In[57]:


Xs_test_prediction = Svm.predict(X_test)
sac = accuracy_score(Xs_test_prediction, Y_test)


# In[58]:


print('Accuracy score on Test Data of support vector machines model: ', sac)


# ## Comparison visualization of the 6 models
# 

# In[59]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
x=["Logistic","Dt","Knn","Randomforest","   Gaussian","Svm"]
y=[lac,dtac,kac,rac,gac,sac]
z=["blue","green","yellow","lime","gold"]
plt.bar(x,y,width=.8,color=z)
plt.xlabel("Algorithms")
plt.ylabel("Accurracy")
plt.show()


# In[60]:


dataset = datasets.load_wine()
X = dataset.data; y = dataset.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)


# In[61]:


model = RandomForestClassifier()
model.fit(X_train, y_train)
filename = "Completed_model.joblib"
joblib.dump(model, filename)


# In[62]:


loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)


# In[ ]:





# In[ ]:




