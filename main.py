# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2021        Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8
from SaneLib import SaneLib
from DataBase import MetaData
import constants as cns
import time
import matplotlib.pyplot as plt

# starting time
start = time.time()

sl = SaneLib({
    'drivername': 'mysql+mysqlconnector',
    'host': 'localhost',
    'port': 3306,
    'username': 'root',
    'password': 'ahjsdnva8s79d',
    'database': 'dbml',
    'query': {'charset': 'utf8'}
})

model_id='covtyp'

sl.db.createView('covtyp_rand', 'select rand() as r, c.* from dbml.covtypall c', materialized=False)
sl.db.createView('covtyp_train', 'select * from covtyp_rand where r < 0.75', materialized=True)
sl.db.createView('covtyp_eval', 'select * from covtyp_rand where r >= 0.75', materialized=True)

table_train='covtyp_train'
table_eval='covtyp_eval'

cat_cols = [
    'Wilderness_Area',
    'Soil_Type',
]

num_cols = [
        'Elevation',
        'Aspect',
        'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am',
        'Hillshade_Noon',
        'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
]

bins = 39

target = 'Cover_Type'

mdh = sl.mdh(model_id)

mdh.descriptive_statistics(table_train, cat_cols,num_cols, bins)

mdh.contingency_table_1d(table_train, cat_cols,num_cols, bins, target)

ranked_columns = mdh.rank_columns_1d()

print(f"-- Runtime of the program is {time.time() - start} seconds")

# Training phase: _qt is trained on 0.8 of table ; _qmt based off of _qt ; _m based off of _qt

mdh.train(
    table_train=table_train,
    catFeatures=['Soil_Type', 'Wilderness_Area'],
    numFeatures=['Elevation','Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points'],
    target=target,
    bins=bins
)

print(f"-- Runtime of the program is {time.time() - start} seconds")

mdh.predict(table_eval='table_eval')
print(f"-- Runtime of the program is {time.time() - start} seconds")

mdh.accuracy()

mdh.update_bayes()
print(f"-- Runtime of the program is {time.time() - start} seconds")

mdh.accuracy()
