import sanelib
from util import timer
from util.database_connection import Database

timer.start()

dtc = sanelib.dtc.DecisionTreeClassifier(db=sanelib.db, dataset='covtyptest2', max_samples=2, max_mutual_inf=0.1)
dtc.train_test_split(ratio=0.01, seed=1, encode=True)
dtc.estimate()
pred = dtc.predict()
acc = 1 - (len(pred.query('Cover_Type != prediction')) / len(pred))
print(acc)

dtc.predict_table(stored_procedure=True)
acc = dtc.score()
print("Accuracy is = {}".format(acc))

timer.end()
