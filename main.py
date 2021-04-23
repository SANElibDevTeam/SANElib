import sanelib
from util import timer
from util.database_connection import Database

timer.start()

numFeatures = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
               'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
               'Horizontal_Distance_To_Fire_Points']
catFeatures = ['Wilderness_Area', 'Soil_Type']

dtc = sanelib.dtc.DecisionTreeClassifier(db=sanelib.db, dataset='covtyptest2')

# dtc.train_test_split(encode=True)

dtc.estimate(numFeatures=numFeatures, catFeatures=catFeatures)
# dtc.visualize_tree()
db = Database(sanelib.db)
test = db.execute_query("select * from covtyptest2_eval", db.engine, True)

test['pred'] = dtc.predict(test)
acc = 1 - (test.query('pred != Cover_Type').shape[0] / test.shape[0])
print(acc)

timer.end()
