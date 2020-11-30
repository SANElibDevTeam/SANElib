# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8

import mysql.connector as mysql
import classifier
import constants as cons

db = mysql.connect(
    host=cons.DB_HOST,
    user=cons.DB_USER,
    passwd=cons.DB_PW,
    database=cons.DB_NAME)

#classifier = SaneProbabilityEstimator(db, 'iris', 'class', 'irismodel')



classifier = classifier.SaneProbabilityEstimator(db, 'table_train', 'Cover_Type', 'covtyptest2')

numFeatures = ["Elevation", "Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Roadways"]
# numFeatures = ["Elevation", "Horizontal_Distance_To_Roadways"]

catFeatures = ["Wilderness_Area", "Soil_Type"]

# classifier.hyperparameters(numFeatures, bins, catFeatures) # Training accuracy =  [Decimal('0.6640')]

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


# There may be a better way in which the user passes parameters to this function - Searching for solutions
# The SaneProbabilityEstimator Object holds all parameters
# perhaps pass the hyperparaemters directly to train method


###sqlGen.getSql_qt(classifier)

#print(sqlGen.getSql_qmt(classifier))
#print(sqlGen.getSql_m(classifier))

allNumFeat = [ "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", \
                "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", \
                "Horizontal_Distance_To_Fire_Points"]
allCatFeat = ["Wilderness_Area", "Soil_Type"]

classifier.rank('table_train', allCatFeat, allNumFeat,  50)

###classifier.train('table_train', catFeatures, numFeatures,  50)


###classifier.predict('table_eval')

### print('Model accuracy = ', classifier.accuracy())

### print('Training accuracy = ', classifier.trainingAccuracy())
