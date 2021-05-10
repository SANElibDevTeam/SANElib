import sanelib
from util import timer
from util.database_connection import Database

timer.start()

dtc = sanelib.dtc.DecisionTreeClassifier(db=sanelib.db, dataset='covtyptest2', max_samples=2)

dtc.train_test_split(ratio=0.02, seed=1, encode=True)

dtc.estimate()

dtc.predict_table()
acc = dtc.score()
print("Accuracy is = {}".format(acc))

timer.end()
