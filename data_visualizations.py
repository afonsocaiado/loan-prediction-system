import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read in all (training) datafiles
# these can be joined later on to use all data for the models
# account = pd.read_csv('data\account.csv',delimiter=";")
card_train = pd.read_csv('data/card_train.csv', delimiter=";")
client = pd.read_csv('data/client.csv', delimiter=";")
disp = pd.read_csv('data/disp.csv', delimiter=";")
district = pd.read_csv('data/district.csv', delimiter=";")
loan_train = pd.read_csv('data/loan_train.csv', delimiter=";")
# trans_train = pd.read_csv('data\trans_train.csv',delimiter=";")

# lets focus on the loan_train dataset for now
dev = loan_train.copy()
dev_lenght = len(dev)
# split up dev data into training and test data and create copies to keep original data
data_split = round((2 / 3) * dev_lenght)
train = (dev.iloc[:data_split, :]).copy()
test = (dev.iloc[data_split + 1:, :]).copy()

print("Lets see the columns to decide which could have colleration:\n", dev.head(), "\n")

# DATA UNDERSTANDING
# mean
average_date = dev['date'].mean()
average_amount = dev['amount'].mean()
average_duration = dev['duration'].mean()
average_payments = dev['payments'].mean()

print("Average of useful columns:\naverage_date =", average_date,
      "\naverage_amount =", average_amount,
      "\naverage_duration =", average_duration,
      "\naverage_payments =", average_payments, "\n")

# print percentage of valid and unvalid loans
print("Percentage of valid(+1) and unvalid(-1) loans and plot:\n", dev['status'].value_counts(normalize=True))

# DATA VISUALIZATION
# barplot of number of valid and unvalid loans
dev['status'].value_counts().plot.bar(title='Status of loans')
plt.show()

print("We can see that there are a lot more valid loans so we will need to do under/over-sampling in the future\n")

print("Lets see the distributions:")
sns.distplot(dev["date"]).set_title("Density of the date values")
plt.show()
sns.distplot(dev["amount"]).set_title("Density of the amount values")
plt.show()
sns.distplot(dev["duration"]).set_title("Density of the duration values")
plt.show()
sns.distplot(dev["payments"]).set_title("Density of the payment values")
plt.show()

# outlier detection
dev["date"].plot.box(title="Date Boxplot")
plt.show()
dev["amount"].plot.box(title="Amount Boxplot")
plt.show()
dev["duration"].plot.box(title="Duration Boxplot")
plt.show()
dev["payments"].plot.box(title="Payments Boxplot")
plt.show()

# line plot with date
dev.plot.line(x="date", y="amount", title="Amount by Date")
plt.show()

# barplot  of average total loan by status
print("Lets see the average loanamount and average monthly loanamount by status:")
dev.groupby('status')['amount'].mean().plot.bar(title='Average loan by status')
plt.show()

# barplot of monthly loan by status
dev.groupby('status')['payments'].mean().plot.bar(title='Average monthly loan by status')
plt.show()

print("We can see that the unpaid loans usually have a higher amount")
