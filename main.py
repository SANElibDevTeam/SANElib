# Visualization_ Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8
import classifier
import constants as cons
import time
import pandas as pd
import jaydebeapi
import sqlite3 as sl

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

# Create directory/file to hold file paths --> .../
path = '/Users/gabestechschulte/covertype.csv'
h2path = '/Users/gabestechschulte/h2/bin/h2-1.4.199.jar'

#classifier = SaneProbabilityEstimator(db, 'iris', 'class', 'irismodel')
#classifier = classifier.SaneProbabilityEstimator(db, 'table_train', 'Cover_Type', 'covtyptest2')
#classifier = classifier.SaneProbabilityEstimator('Cover_Type', 'covtypeH2', path, h2path)
classifier = classifier.SaneProbabilityEstimator(None, 'covtypeH2', 'Cover_Type', None, path, h2path)

allNumFeat = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", \
                "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", \
                "Horizontal_Distance_To_Fire_Points"]
allCatFeat = ["Wilderness_Area", "Soil_Type"]

#classifier.rank('table_train', allCatFeat, allNumFeat,  50)
#classifier.rank('covtyp', allCatFeat, allNumFeat,  50)

#TODO automate attribute selection based on threshold
numFeatures = ["Elevation", "Horizontal_Distance_To_Fire_Points"]
bins = 57
catFeatures = ["Wilderness_Area", "Soil_Type"]

#classifier.train_test_split(1, 0.8)

# Training phase: _qt is trained on 0.8 of table ; _qmt based off of _qt ; _m based off of _qt
classifier.train('SANE_TABLE', catFeatures, bins, numFeatures)

# Pass what feature and the target and what dimension you want to see the density function for
classifier.visual('Elevation', 'Cover_Type')

# Predicting on test set: _qe tested on 0.2 of table ; _qe_ix based off of _qe ; _p ; _p_update
#classifier.predict('covtypeH2_table_eval')

classifier.accuracy()

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start} seconds")

# print('Training accuracy = ', classifier.trainingAccuracy())
