# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8
import constants as cons
import time
from Utils import *
import matplotlib.pyplot as plt
import Database
import Analysis
from pandas import util

# starting time
start = time.time()

db = {
        'drivername': 'mysql+mysqlconnector',
        'host': cons.DB_HOST,
        'port': cons.DB_PORT,
        'username': cons.DB_USER,
        'password': cons.DB_PW,
        'database': cons.DB_NAME,
        'query': {'charset': 'utf8'}
    }


#TODO automate attribute selection based on threshold
numFeatures = ["Elevation", "Horizontal_Distance_To_Fire_Points"]
bins = 57
catFeatures = ["Wilderness_Area", "Soil_Type"]



df= pd.DataFrame({'A' : [1,2,3]})
# Add df instead of already working db with
# engine = Database.Database(dataframe=df).engine
engine = Database.Database(db).engine

ay = Analysis.Analysis(engine=engine,dataset="table_train",target='Cover_Type',seed=1,ratio=0.8,model_id='covtyptest2')
ay.rank("table_train",catFeatures,numFeatures,bins).estimate(catFeatures,bins,numFeatures).visualize1D('Wilderness_Area', 'Covertype').predict('table_eval').accuracy()

# Estimation phase: _qt is estimated on 0.8 of table ; _qmt based off of _qt ; _m based off of _qt
# Predicting on test set: _qe tested on 0.2 of table ; _qe_ix based off of _qe ; _p ; _p_update

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start} seconds")

# print('Training accuracy = ', classifier.trainingAccuracy())
