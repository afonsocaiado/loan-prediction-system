import collections

def create(competition,classifier):
	competition_inputs = competition[['loan_date', 'amount', 'duration', 'payments',
       'frequency', 'account_date', 'region', 'inhabitants',
       'inhabitants < 499', 'inhabitants 500-1999', 'inhabitants 2000-9999',
       'inhabitants >10000', 'no. of cities ', 'ratio of urban inhabitants ',
       'average salary ', 'unemploymant 96',
       'enterpreneurs', 'crimes 96']].values


	competition_prob = classifier.predict_proba(competition_inputs)

	d = {}
	for v in competition.index:
		d[competition["loan_id"][v]] = competition_prob[v][1]
	
	d = collections.OrderedDict(sorted(d.items()))

	submission_file = open('submission_file.csv', 'w')
	submission_file.write("{},{}\n".format("Id","Predicted"))
	for c in d:
    		submission_file.write("{},{}\n".format(c,d[c]) )

	submission_file.close()
