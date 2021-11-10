import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,getopt


def main(argv):
	opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])

	dev = pd.read_csv(args[0],delimiter=";")

	print(len(dev))
	ok = []
	not_ok = []
	for t in dev.index:	
		if(dev["date"][t] > 941000):
			ok.append(dev['loan_id'][t])
		else:
			not_ok.append(dev['loan_id'][t])
	
	print(len(not_ok)/(len(ok)+len(not_ok))*100)
	print(len(ok))
	print(len(not_ok))
	
if __name__ == "__main__":
   main(sys.argv[1:])
