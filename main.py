import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

input_file=pd.read_csv("Dataset/Dataset.csv",low_memory=False)

# print(input_file.info()) #gives information about the file
# print(input_file.head()) #returns first 5 rows
# print(input_file.describe()) 

input_missing=pd.DataFrame({'column':input_file.columns, 'percent_missing':input_file.isnull().sum()*100/len(input_file)})
input_file.drop(input_missing[input_missing['percent_missing']>10].index,axis=1,inplace=True)
print(input_file.info())

#columns=['loan_amnt','term','int_rate','installment','grade','emp_length','home_ownership',' ']

#transforming loan status from categorical to numeric
input_file['loan_status_numeric']=input_file['loan_status'].map({'Charged Off':1,'Fully Paid':0})

#grade distribution
#(input_file['grade'].value_counts().sort_index()/len(input_file)).plot.bar()