import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,getopt


def main(argv):
	opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])

	dev = pd.read_csv(args[0],delimiter=";")
	
	train = dev.copy()
	test = dev.copy()

	test = test.loc[test['date'] < 941000]
	
	train = train.loc[(train['date'] >= 941000)]
	
if __name__ == "__main__":
   main(sys.argv[1:])
