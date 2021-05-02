import time

import pandas as pd

import lib
import sanelib
from util.database import Database

iris_df = pd.read_csv("datasets/iris.csv")
db_connection = {
    "drivername": "sqlite",
    "path": "",
}
db = Database(db_connection=db_connection, dataframe=iris_df, dfname="iris")
kmeans = lib.kmeans.KMeans(db)
# kmeans = sanelib.kmeans

model_names = kmeans.get_model_names()
print(f"models: {model_names}")
# kmeans.drop_model(model_names[-1])
# model = kmeans.load_model(model_names[0])

# tablename = "covtypall"
# feature_names = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]

tablename = "iris"
feature_names = ["sepallength", "sepalwidth", "petallength", "petalwidth"]

# tablename = "mouse"
# feature_names = ["x","y"]

k = 3
k_list = range(2,5)
model_identifier = "example"
normalizations = [None, "min-max", "z-score"]

init_time = time.time()
model = kmeans.create_ideal_model(tablename, feature_names, k_list, model_identifier, normalizations[1])
train_time = time.time()
# model.estimate(max_steps=10)
print(f"Information: {model.get_information()}")
print(f"Initialization: {train_time - init_time} [s]")
print(f"Training: {time.time() - train_time} [s]")


axis_order = [3, 0, 2]
model.visualize(feature_names)
