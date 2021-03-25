# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8
import config as cons
import time
from util import database_connection
import sanelib
import pandas as pd

# starting time
start = time.time()

#TODO automate attribute selection based on threshold
numFeatures = ["Elevation", "Horizontal_Distance_To_Fire_Points"]
bins = 57
catFeatures = ["Wilderness_Area", "Soil_Type"]


df= pd.DataFrame({'A' : [1,2,3]})
# Add df instead of already working db with
# engine = Database.Database(dataframe=df).engine
# engine = database_connection.Database(db).engine

# ay = Analysis.Analysis(engine=engine)
mdh_test = sanelib.mdh
mdh_test.rank("table_train",catFeatures,numFeatures,bins).estimate(catFeatures,bins,numFeatures).visualize1D('Wilderness_Area', 'Covertype').predict('table_eval').accuracy()

# Estimation phase: _qt is estimated on 0.8 of table ; _qmt based off of _qt ; _m based off of _qt
# Predicting on test set: _qe tested on 0.2 of table ; _qe_ix based off of _qe ; _p ; _p_update

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start} seconds")

# print('Training accuracy = ', classifier.trainingAccuracy())
