import pandas as pd
import matplotlib.pyplot as plt

def prep_data():
    #read datasets
    account = pd.read_csv('data/account.csv',na_values = ['NA'],delimiter=";")
    card_train = pd.read_csv('data\card_train.csv',na_values = ['NA'],delimiter=";")
    client = pd.read_csv('data\client.csv',na_values = ['NA'],delimiter=";")
    disp = pd.read_csv('data\disp.csv',na_values = ['NA'],delimiter=";")
    district = pd.read_csv('data\district.csv',na_values = ['NA'],delimiter=";")
    loan_train = pd.read_csv('data\loan_train.csv',na_values = ['NA'],delimiter=";")
    trans_train = pd.read_csv('data/trans_train.csv', dtype={'bank' : 'str'}, na_values = ['NA'],delimiter=";")
    
    #join together loan_train and account and rename the columns with the same name
    account.rename(columns={'date': 'account_date'}, inplace=True)
    loan_train.rename(columns={'date': 'loan_date'}, inplace=True)
    loan_account = pd.merge(loan_train, account, on="account_id")
    
    #join loan_account with district but first remove unuseful columns
    #every name is distinct so we can remove the column
    district.drop('name ', inplace =True, axis=1)
    loan_account_district = pd.merge(loan_account, district, left_on="district_id", right_on="code ")
    
    return loan_account_district


#join disp and card_train
#card_train.drop(['card_id', 'issued'], axis = 1, inplace=True)
#ard_train.rename(columns={'type': 'card_type'}, inplace=True)
#disp.rename(columns={'type': 'disp_type'}, inplace=True)
#disp_card = pd.merge(disp, card_train, on="disp_id")
#disp_card.drop('disp_id', axis = 1, inplace=True)
#print(disp_card.columns)
#disp_card.groupby('account_id')