import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

input_file=pd.read_csv("Dataset/Dataset.csv",low_memory=False,nrows=20000)

print(input_file.info()) #gives information about the file


# Checking for null values before preprocessing
sns.set(rc={'figure.figsize':(10,10)})
sns.set_style('whitegrid')
sns.heatmap(input_file.isnull())
plt.title('Null values visual plot',fontdict={'fontsize': 10})
plt.legend(input_file.isnull())
plt.show()

input_missing=pd.DataFrame({'column':input_file.columns, 'percent_missing':input_file.isnull().sum()*100/len(input_file)})
input_file.drop(input_missing[input_missing['percent_missing']>10].index,axis=1,inplace=True)
print(input_file.isnull().sum())
input_file.drop(['id'],axis=1,inplace=True)
print(input_file.info())

# Checking for null values after null value fields
sns.heatmap(input_file.isnull())
plt.title('Null values visual plot',fontdict={'fontsize': 10})
plt.legend(input_file.isnull())
plt.show()

#transforming loan status from categorical to numeric
input_file['loan_status_numeric']=input_file['loan_status'].map({'Charged Off':1,'Fully Paid':0})

#feature extraction
columns=['loan_amnt','term','int_rate','installment','grade','emp_length','home_ownership','annual_inc',
        'verification_status','loan_status_numeric','purpose','addr_state','dti','delinq_2yrs','fico_range_low',
        'inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc']

input_file=input_file[columns]

# Checking for null values after feature extraction
sns.heatmap(input_file.isnull())
plt.title('Null values visual plot',fontdict={'fontsize': 10})
plt.legend(input_file.isnull())
plt.show()

