import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prep_data(arg):
    """
    

    Parameters
    ----------
    arg : String
        Use train for data prep of training data, use competition for data prep of competition data.

    Returns
    -------
    loan_account_district : Dataframe
        returns joined data.

    """
    if arg == 'train':
        loan = pd.read_csv('../data/loan_train.csv', na_values=['NA'], delimiter=";")
        trans = pd.read_csv('../data/trans_train.csv', dtype={'bank': 'str'}, na_values=['NA'], delimiter=";")
        card = pd.read_csv('../data/card_train.csv', na_values=['NA'], delimiter=";")
    elif arg == 'competition':
        loan = pd.read_csv('../data/loan_test.csv', na_values=['NA'], delimiter=";")
        trans = pd.read_csv('../data/trans_test.csv', dtype={'bank': 'str'}, na_values=['NA'], delimiter=";")
        card = pd.read_csv('../data/card_test.csv', na_values=['NA'], delimiter=";")
    else:
        print("ERROR: invalid argument")

    # read datasets
    account = pd.read_csv('../data/account.csv', na_values=['NA'], delimiter=";")
    card = pd.read_csv('../data/card_train.csv', na_values=['NA'], delimiter=";")
    client = pd.read_csv('../data/client.csv', na_values=['NA'], delimiter=";")
    disp = pd.read_csv('../data/disp.csv', na_values=['NA'], delimiter=";")
    district = pd.read_csv('../data/district.csv', na_values=['NA'], delimiter=";")

    # join together loan_train and account and rename the columns with the same name
    account.rename(columns={'date': 'account_date'}, inplace=True)
    loan.rename(columns={'date': 'loan_date'}, inplace=True)
    loan_account = pd.merge(loan, account, on="account_id")

    # rename columns
    district.rename(columns={'no. of inhabitants': 'inhabitants',
                             'no. of municipalities with inhabitants < 499 ': 'inhabitants < 499',
                             'no. of municipalities with inhabitants 500-1999': 'inhabitants 500-1999',
                             'no. of municipalities with inhabitants 2000-9999 ': 'inhabitants 2000-9999',
                             'no. of municipalities with inhabitants >10000 ': 'inhabitants >10000',
                             "unemploymant rate '95 ": 'unemploymant 95', "unemploymant rate '96 ": 'unemploymant 96',
                             'no. of enterpreneurs per 1000 inhabitants ': 'enterpreneurs',
                             "no. of commited crimes '95 ": 'crimes 95', "no. of commited crimes '96 ": 'crimes 96'},
                    inplace=True)
                    
    # replace question marks by mean value of column
    replace_dict = {'unemploymant 95': {'?': pd.to_numeric(district['unemploymant 95'], errors="coerce").mean()},
                    'crimes 95': {'?': pd.to_numeric(district['crimes 95'], errors="coerce").mean()}}
    district.replace(replace_dict, inplace=True)
    district[['crimes 95', 'unemploymant 95']] = district[['crimes 95', 'unemploymant 95']].apply(pd.to_numeric)

    # join loan_account with district
    loan_account_district = pd.merge(loan_account, district, left_on="district_id", right_on="code ")
    # remove unuseful columns
    loan_account_district.drop(['district_id', 'code ', 'name '], axis=1, inplace=True)
    # replace string values with numerical values for analysis
    freq = {'monthly issuance': 1, "weekly issuance": 2, "issuance after transaction": 3}
    loan_account_district.frequency = [freq[i] for i in loan_account_district.frequency]
    reg = {"south Moravia": 1, "north Moravia": 2, "Prague": 3, "central Bohemia": 4, "east Bohemia": 5,
           "west Bohemia": 6, "south Bohemia": 7, "north Bohemia": 8}
    loan_account_district.region = [reg[i] for i in loan_account_district.region]

    # prepare trans
    trans.drop(['type', 'operation', 'k_symbol', 'bank', 'account', 'amount', 'trans_id'], axis=1, inplace=True)
    trans = trans.groupby('account_id')['balance'].agg([np.min, np.max, np.mean])
    trans = trans.rename(columns={'amin': 'min_balance', 'amax': 'max_balance', 'mean': 'avg_balance'})
    # join trans and loan_account_district
    loan_account_district_trans = pd.merge(loan_account_district, trans, on="account_id")
    # create new columns with different statistics
    loan_account_district_trans['effort'] = loan_account_district_trans['amount'] / loan_account_district_trans[
        'avg_balance']
    loan_account_district_trans['salary_effort'] = loan_account_district_trans['amount'] / loan_account_district_trans[
        'average salary ']
    loan_account_district_trans['monthly_effort'] = loan_account_district_trans['payments'] / \
                                                    loan_account_district_trans['average salary ']

    # prepare to join card
    card.drop(['card_id', 'issued'], axis=1, inplace=True)
    card.rename(columns={'type': 'card_type'}, inplace=True)
    disp.rename(columns={'type': 'disp_type'}, inplace=True)
    disp_card = pd.merge(disp, card, on="disp_id")
    disp_card.drop(['disp_id', 'client_id', 'disp_type'], axis=1, inplace=True)
    cardtypes = {'junior': 1, 'classic': 2, 'gold': 3}
    disp_card.card_type = [cardtypes[i] for i in disp_card.card_type]
    loan_account_district_trans_card = pd.merge(loan_account_district_trans, disp_card, on="account_id", how="left")
    loan_account_district_trans_card['card_type'].fillna(int(0), inplace=True)

    # get male or female and birth date
    disp = pd.read_csv('../data/disp.csv', na_values=['NA'], delimiter=";")
    disp.rename(columns={'type': 'disp_type'}, inplace=True)
    disp_client = pd.merge(disp, loan, on="account_id")
    disp_client.drop(['disp_id', 'account_id', 'loan_date', 'amount', 'duration', 'payments', 'status'], axis=1,
                     inplace=True)
    disp_client = pd.merge(disp_client, client, on="client_id")
    disp_client = disp_client.loc[disp_client['disp_type'] == "OWNER"]
    disp_client['is_male'] = [isMale(n) for n in disp_client.birth_number]
    disp_client.birth_number = [getDate(n) for n in disp_client.birth_number]
    disp_client.drop(['client_id', 'disp_type', 'district_id'], axis=1, inplace=True)
    # print(disp_client)

    # join birth date
    loan_account_district_trans_card = pd.merge(loan_account_district_trans_card, disp_client, on="loan_id")
    # drop account id
    loan_account_district_trans_card.drop('account_id', axis=1, inplace=True)

    return loan_account_district_trans_card


def isMale(n):
    n = str(n)
    m = int(n[2:4])
    if m > 12:
        return 0
    else:
        return 1


def getDate(n):
    n = str(n)
    y = n[0:2]
    m = int(n[2:4])
    d = n[4:6]
    if m > 12:
        m -= 50
    return str(y) + "{:02d}".format(m) + str(d)
