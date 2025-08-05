import numpy as np
import pandas as pd

tid = [1,2,3,4,5,6,7,8,9,10]
refund = ['yes','no','no','yes','no','no','yes','no','no','no']
marital_status = ['single','married','single','married','divorced','married','divorced','single','married','single']
taxable_income = ['125K','100K','70K','120K','95K','60K','220K','85K','75K','90K']
cheat = ['no','no','no','no','yes','no','no','yes','no','yes']
data = {
    'Tid':tid,
    'Refund':refund,
    'Marital Status':marital_status,
    'Taxable Income':taxable_income,
    'Cheat':cheat
}

df = pd.DataFrame(data)
print(df)

print("Row 0:", df.iloc[0])
print("Row 4:", df.iloc[4])
print("Row 7:", df.iloc[7])
print("Row 8:", df.iloc[8])

print(df[3:8])
print()
print(df.iloc[4:9,2:4])

print()
data = pd.read_csv('iris.data.csv')
print(data[:5])
# or 
print()
print(data.head(5))

print()
print(data)

print()
data2 = data.drop(index = 3).drop(data.columns[1], axis = 1)
print(data2)

data = {
    'Employee_ID':[101,102,103,104,105],
    'Name':['Alice','Bob','Charlie','Diana','Edward'],
    'Department':['HR','IT','IT','Marketing','Sales'],
    'Age':[29,34,41,28,38],
    'Salary':[50000,70000,65000,55000,60000],
    'Years of Experience':[4,8,10,3,12],
    'Joining Date':['2020-03-15','2017-07-19','2013-06-01','2021-02-10','2010-11-25'],
    'Gender':['Female','Male','Male','Female','Male'],
    'Bonus':[5000,7000,6000,4500,5000],
    'Rating':[4.5,4.0,3.8,4.7,3.5]
}

df = pd.DataFrame(data)
print(df)

print()
print(df.shape)
print(df.info())
print(df.describe())
print(df.head(5))
print(df.tail(3))

print()
print(np.mean(df['Salary']))
print(np.sum(df['Bonus']))
print(np.min(df['Age']))
print(np.max(df['Rating']))

print()
sorted_df = df.sort_values(by='Salary', ascending=False)
print(sorted_df)

print()
performance = []
for i in df['Rating']:
    if i >= 4.5:
        performance.append('Excellent')
    elif i >= 4.0:
        performance.append('Good')
    else:
        performance.append('Average')
    
df['Performance Rating'] = performance
print(df)

print(df.isna())

df = df.rename(columns={'Employee_ID':'ID'})
print()
print(df)

print()
print(df[df['Years of Experience'] > 5])

print()
print(df[df['Department'] == 'IT'])

print()
df['Tax'] = 0.1*df['Salary']
print(df)

df.to_csv('final_df.csv')