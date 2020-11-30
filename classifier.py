import sqlTemplates as sql
from jinja2 import Template
import pandas as pd

class SaneProbabilityEstimator:

    def __init__(self, conn, table_train, target=None, model_id='table_name'):
        """
        This method takes the most commonly used parameters from the user during initialization of the classifier
        - conn = database connection (format: variable)
        - target = target variable (what you are wanting to predict) (format: str)
        - table_name = name of the table that is in the database (format: str)
        """
        self.connection = conn
        self.table_train = table_train
        self.model_id = model_id

        if target is None:
            col = self.executeQuery('If last column (DESC), else first column (ASC)', '''
                SELECT
                COLUMN_NAME
                ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{}'  #table_train
                ORDER BY ORDINAL_POSITION DESC
                LIMIT 1;
                '''.format(table_train))

            # ".fetchall" returns a list of tuples so this
            # for loop manually parses for the last column in table
            for index, tuple in enumerate(col):
                self.target = tuple[col]
        else:
            self.target = target
        cursor = self.connection.cursor()
        cursor.execute("rollback;")
        cursor.close()



# TODO use SQL alchemy or similar framework that works for all SQL DB types
    def execute(self, desc, query):
        cursor = self.connection.cursor()
        print(desc + '\nQuery: ' + query)
        cursor.execute(query)
        cursor.close()
        print('OK: ' + desc)
        print()

    def executeQuery(self, desc, query):
        cursor = self.connection.cursor()
        print('Query: ' + query)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        print('OK: ' + desc)
        print()
        return results

    def materializedView(self, desc, tablename, query):
        self.execute('Dropping table ' + tablename, '''
            drop table if exists {}'''
                .format(tablename))
        self.execute(desc, '''
            create table {} as '''
                .format(tablename) + query)

   # TODO develop an algorithm to optimize the hyper parameters
    #  for n buckets: Idea Nr. 1: Linear, straight forward. first sort the features according to 1D prediction accuracy;
    # Start with the first in the list;
    # for each feature, increase n-buckets until convergence;
    # Add next best feature, and so on, until convergence
    # Idea Nr. 2: Evolutionary. again sort by 1D accuracy. Start with the first in the list;
    # Always add the feature that brings the highest performance gain
    # Always increase the bucket number of the feature that brings the highest performance gain.
    # Until convergence => find method that scales well for large data set with acceptable performance
    # Idea 3: linear in feature list, but evolutionary in n-buckets
    # TODO idea: optimize feature list using "random restaurant" simliar to random forest, but using decision "tables" instead of "trees" <-- advanced stuff


    def trainingAccuracy(self):
        """
         This function evaluates the hyperparameters quickly on the training set.
         possible parameters: size of internal modeling / validation split to make it faster
        """
        
        self.train('''(select * from {} where rand() < 0.8) as t'''
                     .format(self.table_train))
        self.predict('''(select * from {} where rand() >= 0.2) as t'''
                .format(self.table_train))
        return self.accuracy()


    def train(self):
        self.train(self.table_train)


    def train(self, table_train, catFeatures, numFeatures, bins  ):
        """
        1.) Need to be feeding the train set (0.8) of original table into this --> training phase
        # TODO No this test train split should be in a separate method
        # Here you take one training table as argument that already exists.
        # In
        2.) Then, invoking the predict method on the test set (0.2) of the original table --> prediction phase on new data
        # No in the predict method you take the actual eval table as argument, not 20% of the train table. We'll have to talk about this.
        
        This function is the training phase:
        - input data is training set
        - the input data table is quantized (equal size) and indexed.
        - This quantized index then represents an in-database model for probability estimation

        This function sets the hyperparameters
        - features to estimate the probability of the target
        - features = right now it is just a string, but I think we may want to explore
                     having the user put the feature(s) into a numpy array. Similiar to
                     how the scikit-learn ML algorithms want them. (format:str)
        - Bins/buckets = # of bins
        """

        # TODO do the same for categorical data types, string etc.
        if numFeatures is None: # TODO here you should take only numerical data types
            allColumns = self.executeQuery('Querying for all of the columns', '''
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{}';  #table_train
            '''.format(self.table_train))

            self.numFeatures = [element for index, tuple in enumerate(allColumns) for index, element in enumerate(tuple)]
        else:
            self.numFeatures = numFeatures

        self.bins = bins
        self.catFeatures = catFeatures


        # TODO Generate queries using n features x1, x2, ..., xn; differentiate between numerical and categorical

        self.materializedView(
            'Quantization of training table',
            self.model_id + '_qt',
            Template(sql.tmplt['_qt']).render(input=self))
        self.materializedView(
            'Quantization metadata for training table',
            self.model_id + '_qmt',
            Template(sql.tmplt['_qmt']).render(input=self))
        self.materializedView(
            'Computing predictive model as contingency table',
            self.model_id + '_m',
            Template(sql.tmplt['_m']).render(input=self))

    def predict(self, table_eval):  # table_eval is the test set
        """
        This function estimates the probabilities for the evaluation data
        """
        
        self.table_eval = table_eval
        
        self.materializedView(
            'Quantization metadata for evaluation table',
            self.model_id + '_qe',
            Template(sql.tmplt['_qe']).render(input=self)) ## generate SQL using Jinja 2 template
        self.execute('Creating index _qe ',
            Template(sql.tmplt['_qe_ix']).render(input=self))
        self.materializedView(
            'Class prediction for evaluation dataset',
            self.model_id + '_p',
            Template(sql.tmplt['_p']).render(input=self))  ## generate SQL using Jinja 2 template
        self.execute('Updating prediction with default prediction for null predictions',
            Template(sql.tmplt['_p_update']).render(input=self))

    def accuracy(self):
        """
        Computing the accuracy of the model and returning the results to the user
        """

        results = self.executeQuery('Computing evaluation accuracy', '''
            select
            count(distinct id) as  cases,
            sum(case when e.y = p.y_ then 1 else 0 end) as tp,
            sum(case when e.y = p.y_ then 1 else 0 end) /  count(distinct id)  as accuracy
            from {}_qe e 
            left outer join {}_p p using(id);
                '''.format(
                    self.model_id,
                    self.model_id))

        df = pd.DataFrame(results)
        df.columns = ['Total', 'TP', 'Accuracy']
        print(df)

    def rank(self, table_train, catFeatures, numFeatures, bins):
        self.numFeatures = numFeatures
        self.bins = bins
        self.catFeatures = catFeatures
        self.materializedView(
            'Computing 1d contingecies with target',
            self.model_id + '_m1d',
            Template(sql.tmplt['_m1d']).render(input=self))
        results = self.executeQuery('Computing mutual information with target',
            Template(sql.tmplt["_m1d_mi"]).render(input=self))
        df = pd.DataFrame(results)
        df.columns = ['f', 'mi']
        print(df)
