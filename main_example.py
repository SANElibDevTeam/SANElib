import pandas as pd

import lib
from util.database_connection import Database

iris_df = pd.read_csv("datasets/iris.csv")
db = Database(dataframe=iris_df, dfname="iris")
kmeans = lib.kmeans.KMeans(db)

tablename = "iris"
feature_names = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
k = 3
model_identifier = "example"

model = kmeans.create_model(tablename, feature_names, k, model_identifier)
print(model.get_information())
