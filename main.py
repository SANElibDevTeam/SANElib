# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2021        Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8
import MDH
import constants as cns
import time
import matplotlib.pyplot as plt

# starting time
start = time.time()

db = {
        'drivername': 'mysql+mysqlconnector',
        'host': cns.DB_HOST,
        'port': cns.DB_PORT,
        'username': cns.DB_USER,
        'password': cns.DB_PW,
        'database': cns.DB_NAME,
        'query': {'charset': 'utf8'}
    }

classifier = MDH.SaneProbabilityEstimator(db, 'iris', 'class', bins=4)

classifier.rank()

print(f"Runtime of the program is { time.time() - start} seconds")

#TODO automate attribute selection based on threshold

#TODO subsample
classifier.train_test_split(1, 0.8)

# Training phase: _qt is trained on 0.8 of table ; _qmt based off of _qt ; _m based off of _qt
classifier.train(catFeatures=[], numFeatures = ['petalwidth', 'petallength', 'sepallength'])
print(f"Runtime of the program is {time.time() - start} seconds")

# Visualization methods
#classifier.visualize1D('Wilderness_Area', 'Covertype')
#classifier.visualize2D('Elevation', 'Wilderness_Area', 'CoverType')

# Predicting on test set: _qe tested on 0.2 of table ; _qe_ix based off of _qe ; _p ; _p_update
classifier.predict()
print(f"Runtime of the program is {time.time() - start} seconds")

classifier.accuracy()


# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {time.time() - start} seconds")

# print('Training accuracy = ', classifier.trainingAccuracy())
