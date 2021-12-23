import collections

def create(competition,classifier):
	competition_inputs = competition.drop(['status','loan_id'], axis = 1)

    #predict_proba() returns 2Darray of probabilities of classes. We have classes -1: unvalid and 1:valid. We want the class with the unvalid loans
    #using classifier.classes_ we can see the order off classes that predict_proba() returns. We can see that the -1 class is the first element
	competition_prob = classifier.predict_proba(competition_inputs)

	d = {}
	for v in competition.index:
		d[competition["loan_id"][v]] = round(competition_prob[v][0],1)
	
	d = collections.OrderedDict(sorted(d.items()))

	submission_file = open('submission_file.csv', 'w')
	submission_file.write("{},{}\n".format("Id","Predicted"))
	for c in d:
    		submission_file.write("{},{}\n".format(c,d[c]) )

	submission_file.close()
