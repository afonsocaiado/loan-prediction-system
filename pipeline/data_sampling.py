import pandas as pd
import sys,getopt
import random


random.seed(10)

def sampling(train):

	positive = train.copy()
	negative = train.copy()
	
	positive = negative.loc[negative["status"] == 1]
	negative = negative.loc[negative["status"] == -1]
	
	
	while(len(positive) != len(negative)):
		if(len(positive) > len(negative)):
			positive = randomRemove(positive)
		elif(len(positive) > len(negative)):
			negative = randomRemove(negative)
	
	return positive.append(negative)
	
def randomRemove(df):
	return df.drop(df.index[random.randint(0,len(df)-1)])
	
