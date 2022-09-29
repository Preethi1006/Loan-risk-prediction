import pandas as pd
import decimal
from math import sqrt
from random import randrange
from random import seed

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = decimal.Decimal(0)
	for i in range(len(row1)-1):
		distance += (decimal.Decimal(row1[i]) - decimal.Decimal(row2[i]))**decimal.Decimal(2)
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		sum_error = decimal.Decimal(0)
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				sum_error += decimal.Decimal(error)**decimal.Decimal(2)
				if bmu[-1] <= row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
	return codebooks

# Test the training function
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
decimal.getcontext().prec=100
input_file=pd.read_csv("Dataset/Dataset.csv",low_memory=False,nrows=2000)

# print(input_file.info())

input_missing=pd.DataFrame({'column':input_file.columns, 'percent_missing':input_file.isnull().sum()*100/len(input_file)})
input_file.drop(input_missing[input_missing['percent_missing']>10].index,axis=1,inplace=True)
# print(input_file.isnull().sum())
# print(input_file.info())
# print(input_file.loan_status.value_counts())

input_file = input_file.dropna(axis = 0).reset_index(drop = True)
# print(input_file.info())

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


input_file['term_years'] = input_file['term'].apply(lambda x : x[0:3])
input_file['term_years'] = input_file.term_years.astype('Int64')
input_file['term_years'] = input_file['term_years']/12
input_file['term_years'] = input_file.term_years.astype('Int64')
input_file['loan_amnt'] = input_file['loan_amnt']/7000
input_file['int_rate'] = input_file['int_rate']/5
input_file['installment'] = input_file['installment']/250
input_file['risk_rate'] = input_file['risk_rate']/5
input_file['annual_inc'] = input_file['annual_inc']/800000
input_file['fico_range_low'] = input_file['fico_range_low']/150
input_file['dti'] = input_file['dti']/27
input_file['open_acc'] = input_file['open_acc']/13
input_file['pub_rec'] = input_file['pub_rec']/5
input_file['revol_bal'] = input_file['revol_bal']/200000
input_file['total_acc'] = input_file['total_acc']/20
input_file['tot_cur_bal'] = input_file['tot_cur_bal']/800000
# print(input_file['term'],input_file['term_years'])

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

columns=['loan_amnt','term_years','int_rate','installment','risk_rate','emp_length_numeric','annual_inc','dti','delinq_2yrs','fico_range_low',
        'inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','tot_cur_bal','loan_status']

# columns=['loan_amnt','term_years','int_rate','installment','risk_rate','emp_length_numeric','home_ownership','annual_inc',
#         'verification_status_numeric','loan_status','dti','delinq_2yrs','fico_range_low',
#         'inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','tot_cur_bal']
input_file=input_file[columns]
print(input_file.info())
inputfile=input_file.to_numpy()
learn_rate = 0.3
n_epochs = 20
n_codebooks = 20
codebooks = train_codebooks(inputfile, n_codebooks, learn_rate, n_epochs)
print('Codebooks: %s' % codebooks)