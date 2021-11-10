import collections

def create(competition,decision_tree_classifier):
	competition_inputs = competition[['date', 'amount',
                         'duration', 'payments']].values


	competition_prob = decision_tree_classifier.predict_proba(competition_inputs)

	d = {}
	for v in competition.index:
		d[competition["loan_id"][v]] = competition_prob[v][0]
	
	d = collections.OrderedDict(sorted(d.items()))

	submission_file = open('submission_file.csv', 'w')
	submission_file.write("{},{}\n".format("Id","Predicted"))
	for c in d:
    		submission_file.write("{},{}\n".format(c,d[c]) )

	submission_file.close()
