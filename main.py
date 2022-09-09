import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

input_file=pd.read_csv("Dataset/Dataset.csv",low_memory=True,nrows=30000)

print(input_file.info()) #gives information about the file

nul = input_file.isnull().mean().sort_values()
nul = nul[nul>0.1]

nul_col = nul.sort_values(ascending = False).index
data = input_file.drop(nul_col, axis = 1)
data = data.dropna(axis = 0).reset_index(drop = True)
print(data.info())

obj_categorical = [feature for feature in data.columns if data[feature].dtype == "O"]
print(obj_categorical)

for value in obj_categorical:
    print(value)
    print(data[value].nunique())

data.drop(['id', 'url'], axis = 1, inplace = True)
data.drop(['sub_grade', 'emp_title'], axis = 1, inplace = True)
data.drop(['title', 'zip_code'], axis = 1, inplace = True)

date_col = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']

for value in date_col:
    data[value + '_month'] = data[value].apply(lambda x : x[0:3])
    data[value + '_year'] = data[value].apply(lambda x : x[-4: ])

data.drop(date_col, axis = 1, inplace = True)

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for column in date_col:
    data[column + '_month'] = data[column + '_month'].apply(lambda x : month_order.index(x))
print(data.info())

for column in data.columns:
    try:
        data[column] = data[column].astype(float)
        
    except:
        pass
print(data.info())

replace_status = {"Fully Paid":"Paid",
             "Current": "Paid",
             "Charged Off": "Default",
              "Does not meet the credit policy. Status:Charged Off":"Default",
              "Does not meet the credit policy. Status:Charged Off":"Default",
              "Does not meet the credit policy. Status:Fully Paid":"Paid",
              "Late (31-120 days)":"Late",
              "Late (16-30 days)":"Late",
              "In Grace Period":"Late",
              "Default":"Default"
             }
data["loan_status"] = data["loan_status"].replace(replace_status)
data = data[ (data["loan_status"]== "Paid") | (data["loan_status"]== "Default")]


print(data.info())