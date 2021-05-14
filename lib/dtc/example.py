import time
from pathlib import Path
import lib
import pandas as pd

from util.database import Database


def run_iris_example():
    iris_df = pd.read_csv("example_datasets/iris.csv")
    db_connection = {
        "drivername": "sqlite",
        "database": "iris",
        "path": str(Path.home())+'/iris.db',
    }
    init_time = time.time()
    db = Database(db_connection=db_connection, dataframe=iris_df, dfname="iris")
    dtc = lib.DecisionTreeClassifier.DecisionTreeClassifier(db)
    dtc.initialize(dataset='iris')
    dtc.train_test_split()
    dtc.estimate()
    dtc.predict_table()
    acc = dtc.score()
    print("\nDecision Tree Visualized:\n")
    dtc.visualize_tree()
    print("Accuracy of Iris Dataset with 80% training data and default stopping criteria is: {}".format(float(acc)))
    print("Process took {} seconds".format(time.time() - init_time))


def run_covertype_example():
    covertype_df = pd.read_csv("example_datasets/covertype.csv")
    db_connection = {
        "drivername": "sqlite",
        "database": "covertype",
        "path": str(Path.home())+"/covertype.db",
    }
    init_time = time.time()
    db = Database(db_connection=db_connection, dataframe=covertype_df, dfname="covertype")
    dtc = lib.DecisionTreeClassifier.DecisionTreeClassifier(db)
    dtc.initialize(dataset='covertype')
    dtc.train_test_split(ratio=0.2, seed=1, encode=True)
    dtc.estimate(max_samples=2, max_mutual_inf=0.1)
    dtc.predict_table()
    acc = dtc.score()
    print("Accuracy of Covertype Dataset with 20% training data and stopping criteria by .1 mutual information is: {}".format(float(acc)))
    print("Process took {} seconds".format(time.time() - init_time))
