# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8

import mysql.connector as mysql
import classifier

db = mysql.connect(
    host="localhost",
    user="root",
    passwd="ahjsdnva8s79d",
    database='dbml')

#classifier = SaneProbabilityEstimator(db, 'iris', 'class', 'irismodel')

classifier = classifier.SaneProbabilityEstimator(db, 'table_train', 'Cover_Type', 'covtyptest2')

# numFeatures = ["Elevation", "Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Roadways"]
numFeatures = ["Elevation", "Horizontal_Distance_To_Roadways"]
bins = [50, 50, 50]
catFeatures = ["Wilderness_Area", "Soil_Type"]

classifier.hyperparameters(numFeatures, bins, catFeatures) # Training accuracy =  [Decimal('0.6640')]

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

classifier.train('table_train')

# Would like to have this function be more like the scikit framework - I.e: classifer.predict(y_test) - Searching for solutions
#Input: name of table with same structure as table_train
classifier.predict('table_eval')

# The print statement of this function needs cleaned up - Working on it

print('Model accuracy = ', classifier.accuracy())

###

# print('Training accuracy = ', classifier.trainingAccuracy())
