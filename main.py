# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8
import classifier
import constants as cons

db = {
        'drivername': 'mysql+mysqlconnector',
        'host': cons.DB_HOST,
        'port': cons.DB_PORT,
        'username': cons.DB_USER,
        'password': cons.DB_PW,
        'database': cons.DB_NAME,
        'query': {'charset': 'utf8'}
    }

#classifier = SaneProbabilityEstimator(db, 'iris', 'class', 'irismodel')
classifier = classifier.SaneProbabilityEstimator(db, 'table_train', 'Cover_Type', 'covtyptest2')

# numFeatures = ["Elevation", "Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Roadways"]
numFeatures = ["Elevation", "Horizontal_Distance_To_Roadways"]
bins = [50, 50, 50]
catFeatures = ["Wilderness_Area", "Soil_Type"]


# Pass hyperparameters directly to train method now --> Ill leave these up as a hyperparameter reference
# classifier.hyperparameters('Aspect', 50) # Training accuracy =  [Decimal('0.4935')]
# classifier.hyperparameters('Slope', 50) # Training accuracy =  [Decimal('0.5085')]
# classifier.hyperparameters('`Horizontal_Distance_To_Hydrology`', 50) # Training accuracy =  [Decimal('0.4985')]
# classifier.hyperparameters('`Vertical_Distance_To_Hydrology`', 50) # Training accuracy =  [Decimal('0.4749')]
# classifier.hyperparameters('`Horizontal_Distance_To_Roadways`', 50) # Training accuracy =  [Decimal('0.4925')]
# classifier.hyperparameters('`Hillshade_9am`', 50) # Training accuracy =  [Decimal('0.4790')]
# classifier.hyperparameters('`Hillshade_Noon`', 50) # Training accuracy =  [Decimal('0.4860')]
# Training accuracy =  [Decimal('0.4882')]
# classifier.hyperparameters('Horizontal_Distance_To_Fire_Points', 9) # Training accuracy =  [Decimal('0.4892')]
# classifier.hyperparameters('`sepalwidth`', 50)


###sqlGen.getSql_qt(classifier)
#print(sqlGen.getSql_qmt(classifier))
#print(sqlGen.getSql_m(classifier))

# Training phase: _qt is trained on 0.8 of table ; _qmt based off of _qt ; _m based off of _qt
classifier.train('covtyptest2_train', catFeatures, bins, 1, 0.8, numFeatures)
# Predicting on test set: _qe tested on 0.2 of table ; _qe_ix based off of _qe ; _p ; _p_update
classifier.predict('covtyptest2_test')

classifier.accuracy()

# print('Training accuracy = ', classifier.trainingAccuracy())
