import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

input_file=pd.read_csv("Dataset/Dataset.csv",low_memory=False,nrows=20000)

print(input_file.info())

input_missing=pd.DataFrame({'column':input_file.columns, 'percent_missing':input_file.isnull().sum()*100/len(input_file)})
input_file.drop(input_missing[input_missing['percent_missing']>10].index,axis=1,inplace=True)
print(input_file.isnull().sum())
print(input_file.info())
print(input_file.loan_status.value_counts())

input_file = input_file.dropna(axis = 0).reset_index(drop = True)
print(input_file.info())

risk_rate_values={'A':[3.41,3.97,4.51,5.14,5.76],
                     'B':[8.28,8.97,9.66,10.35,11.03],
                     'C':[12.25,13.19,14.07,14.9,15.69],
                     'D':[17.57,19.5,22,24.6,25.94],
                     'E':[23.85,23.87,23.9,23.92,23.95],
                     'F':[24.3,24.64,25.12,25.6,25.7],
                     'G':[25.74,25.79,25.84,25.89,25.94]}

ir_row = input_file['sub_grade'].apply(lambda x : x[0:1])
ir_col = input_file['sub_grade'].apply(lambda x : x[1:2])
ir_col = [eval(k) for k in ir_col]
temp = []
for i in range(len(input_file)):
        temp.insert(i,risk_rate_values[ir_row[i]][ir_col[i]-1])

input_file = input_file.assign(risk_rate=temp)

print(input_file['risk_rate'],input_file['sub_grade'])
input_file['term_years'] = input_file['term'].apply(lambda x : x[0:3])
input_file['term_years'] = input_file.term_years.astype('Int64')
input_file['term_years'] = input_file['term_years']/12
input_file['term_years'] = input_file.term_years.astype('Int64')

input_file['emp_length_numeric']=input_file['emp_length'].map({'10+ years':1,'< 1 year':0.05,'2 years':0.2,'3 years':0.3,'1 year':0.1,'5 years':0.5,'6 years':0.6,'4 years':0.4,'8 years':0.8,'9 years':0.9,'7 years':0.7})
input_file['verification_status_numeric']=input_file['verification_status'].map({'Not Verified':0,'Source Verified':1,'Verified':0.5})
input_file['home_ownership']=input_file['home_ownership'].map({'OWN':1,'RENT':0,'MORTGAGE':0.5})


replace_status = {'Fully Paid':1,
                  'Current': 1,
                  'Charged Off': 0,
                  'Does not meet the credit policy. Status:Charged Off':0,
                  'Does not meet the credit policy. Status:Charged Off':0,
                  'Does not meet the credit policy. Status:Fully Paid':1,
                  'Late (31-120 days)':-1,
                  'Late (16-30 days)':-1,
                  'In Grace Period':-1,
                  'Default':0
                }
input_file['loan_status'] = input_file['loan_status'].replace(replace_status)
input_file = input_file[ (input_file['loan_status']== 1) | (input_file['loan_status']== 0)]

columns=['loan_amnt','term_years','int_rate','installment','risk_rate','emp_length_numeric','home_ownership','annual_inc',
        'verification_status_numeric','loan_status','dti','delinq_2yrs','fico_range_low',
        'inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','tot_cur_bal']

input_file=input_file[columns]
print(input_file.info())
print(input_file.head())
print(input_file['home_ownership'])
print(input_file.home_ownership.value_counts())


# input_file['ir_row'] = input_file['sub_grade'].apply(lambda x : x[0:1])
# input_file['ir_col'] = input_file['sub_grade'].apply(lambda x : x[1:2])
# input_file['ir_col'] = input_file.ir_col.astype('Int64')
# ir = pd.Series(risk_rate_values[input_file['ir_row']][input_file['ir_col']-1],index=input_file.index)
# print(ir)
# input_file['risk_rate']=input_file['loan_amnt']
# print("LENGTH OF THE FILE" + str(len(input_file)))
# for i in range(len(input_file)):
#         input_file['risk_rate'][i] = risk_rate_values[input_file['ir_row'][i]][input_file['ir_col'][i]-1]


# input_file.drop(input_file.index[input_file['loan_status'] == 'Current'], inplace=True)

# input_file.loc[input_file['emp_length'].isnull(), 'emp_length'] = '10+ years'

# #transforming loan status from categorical to numeric
# input_file['loan_status_numeric']=input_file['loan_status'].map({'Charged Off':1,'Fully Paid':0,'Late (31-120 days)':0,'Late (16-30 days)':0,'In Grace Period':1})

# date_col=['term']
# for value in date_col:
#     input_file[value + '_month'] = input_file[value].apply(lambda x : x[0:3])

# input_file['term_years']=input_file['term'].map({'36 months':3,'60 months':5})
# print(input_file['term'])
# #feature extraction
# columns=['loan_amnt','term_years','int_rate','installment','grade','emp_length','home_ownership','annual_inc',
#         'verification_status','loan_status_numeric','purpose','addr_state','dti','delinq_2yrs','fico_range_low',
#         'inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc']

# input_file=input_file[columns]

#print(input_file['loan_status_numeric'].value_counts())

#print(input_file.term.value_counts())


#input_file.drop(['term'], axis = 1, inplace = True)
# print(input_file['term_years'])