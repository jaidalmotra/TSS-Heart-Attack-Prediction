# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

train_df=pd.read_csv('/kaggle/input/thapar-summer-school-basic-classification/Train_dataset.csv')
train_df.head()

test_df=pd.read_csv('/kaggle/input/thapar-summer-school-basic-classification/test_dataset.csv')
test_df.head()

train_df.info()

train_df.isnull().sum()

# EDA
    #From above we can see that whole data is either integer or float type

train_df.head()

# We don't need patient_id column becase it has no impact on the output
train_df.drop('patient_id', axis=1, inplace=True)

train_df.head(1)

train_df['age'].value_counts()

plt.figure(figsize=(10,5))
c= train_df.corr()
sns.heatmap(c,annot=True)
# visualising age vs output
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(data=train_df, x="age", y="output",hue='sex',ax=ax)

# visualization slp vs output
sns.barplot(x='slp',y='output',hue='sex',data=train_df)

# visualization thalachh vs output
sns.boxplot(data=train_df, x="thalachh")

X=train_df.drop('output', axis=1)
X.head()

y=train_df['output']
y.head()

# Comparing models using Pycaret
#pip install pycaret &> /dev/null
print ("Pycaret installed sucessfully!!")

from pycaret.classification import *
s = setup(data=train_df, target='output')
cm = compare_models()

setup(data=train_df, target='output',
      remove_outliers = True, outliers_threshold = 0.05,
      normalize = True, normalize_method = 'zscore')
cm = compare_models()

setup(data=train_df,target='output',
      remove_outliers = True, outliers_threshold = 0.05,
      normalize = True, normalize_method = 'zscore',
      transformation = True, transformation_method = 'yeo-johnson')
cm = compare_models()

ada_model=create_model('ada', return_train_score=True)

test_df.head()

test_data=test_df.drop('patient_id', axis=1)
test_data.head()

newPredictions=predict_model(ada_model, data=test_data)
newPredictions

newPredictions.to_csv("NewPredictions.csv", index=False)

sm = save_model(ada_model, 'ada_Model_File')

plot_model(ada_model, plot='confusion_matrix')

plot_model(ada_model, plot='parameter')

plot_model(ada_model, plot='feature')

y_predict=newPredictions['prediction_label']
y_predict

submission = pd.DataFrame({
        "patient_id": test_df["patient_id"],
        "output": y_predict
    })
submission.to_csv('submission.csv', index=False)
