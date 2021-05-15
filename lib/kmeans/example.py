
import lib
import pandas as pd
import sanelib
from util.database import Database


def run_iris_example():
    iris_df = pd.read_csv("example_datasets/iris.csv")
    db_connection = {
        "drivername": "sqlite",
        "database": "",
        "path": "",
    }
    db = Database(db_connection=db_connection, dataframe=iris_df, dfname="iris")
    kmeans = lib.kmeans.KMeans(db)

    tablename = "iris"
    feature_names = ["sepallength", "sepalwidth", "petallength", "petalwidth"]

    k = 3
    model_identifier = "example"
    normalizations = [None, "min-max", "z-score"]

    model = kmeans.create_model(tablename, feature_names, k, model_identifier, normalizations[0])
    
    model.estimate(max_steps=30)

    print(f"Information: {model.get_information()}")

    axis_order = [3, 0, 2]
    model.visualize(feature_names, axis_order)
