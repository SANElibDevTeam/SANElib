import pandas as pd

import lib
import sanelib
from util.database_connection import Database

# iris_df = pd.read_csv("datasets/iris.csv")
# db = Database(dataframe=iris_df, dfname="iris")
# kmeans = lib.kmeans.KMeans(db)
kmeans = sanelib.kmeans

tablename = "iris"
feature_names = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
k = 3
model_identifier = "example"

model = kmeans.create_model(tablename, feature_names, k, model_identifier)
print(f"before clustering: {model.get_information()}")
model.cluster(max_steps=10)
print(f"after clustering: {model.get_information()}")
