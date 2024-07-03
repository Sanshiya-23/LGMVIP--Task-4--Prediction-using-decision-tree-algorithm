#!/usr/bin/env python
# coding: utf-8

# # LGMVIP- Task 4- Prediction using decision tree algorithm

# # Step 1 : Import Libraries

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# # Step 2: Load the Dataset

# In[3]:


#Load the dataset
file_path = r'C:\Users\SANSHIYA\Downloads\Iris.csv'  # Update the path to the location of your Iris.csv file
iris_data = pd.read_csv(file_path)


# # Step 3: Prepare the Data

# In[4]:


#Prepare the data
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']


# # Step 4: Split the Data and Create and Train the Decision Tree Classifier

# In[5]:


#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# # Step 5: Plot the Decision Tree

# In[6]:


# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_, rounded=True)
plt.title("Decision Tree Classifier for Iris Dataset")
plt.show()

