import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read in all (training) datafiles
#these can be joined later on to use all data for the models
#account = pd.read_csv('data\account.csv',delimiter=";")
card_train = pd.read_csv('data\card_train.csv',delimiter=";")
client = pd.read_csv('data\client.csv',delimiter=";")
disp = pd.read_csv('data\disp.csv',delimiter=";")
district = pd.read_csv('data\district.csv',delimiter=";")
loan_train = pd.read_csv('data\loan_train.csv',delimiter=";")
#trans_train = pd.read_csv('data\trans_train.csv',delimiter=";")

#lets focus at the loan_train dataset for now
dev = loan_train.copy()
dev_lenght = len(dev)
#split up dev data into training and test data and create copies to keep original data
data_split = round((2/3)*dev_lenght)
train = (dev.iloc[:data_split,:]).copy()
test = (dev.iloc[data_split+1:,:]).copy()

print("Lets see the columns to decide which could have colleration:\n", dev.head(),"\n")

average_date = dev['date'].mean()
average_amount = dev['amount'].mean()
average_duration = dev['duration'].mean()
average_payments = dev['payments'].mean()
print("average of useful columns:\naverage_date =", average_date,"\naverage_amount =", average_amount,
      "\naverage_duration =", average_duration, "\naverage_payments =", average_payments, "\n")

print("lets see the distributions:")
sns.distplot(dev["date"])
plt.show()
sns.distplot(dev["amount"])
plt.show()
sns.distplot(dev["duration"])
plt.show()
sns.distplot(dev["payments"])
plt.show()

#outlier detection
plt.show()
dev["date"].plot.box(figsize=(16,5))
plt.show()
dev["amount"].plot.box(figsize=(16,5))
plt.show()
dev["duration"].plot.box(figsize=(16,5))
plt.show()
dev["payments"].plot.box(figsize=(16,5))

#print percentage of valid and unvalid loans
print("percentage of valid(+1) and unvalid(-1) loans and plot:\n",dev['status'].value_counts(normalize=True))

#barplot of number of valid and unvalid loans
dev['status'].value_counts().plot.bar(title = 'Status of loans')
plt.show()
print("We can see that there are a lot more valid loans so we will need to do under/over-fitting in the future\n")

#barplot  of average total loan by status
print("Lets see the average loanamount and average monthly loanamount by status:")
dev.groupby('status')['amount'].mean().plot.bar(title = 'average loan by status')
plt.show()

#barplot of monthly loan by status
dev.groupby('status')['payments'].mean().plot.bar(title = 'average monthly loan by status')
plt.show()
    
print("We can see that the unpaid loans usually have a higher amounts")