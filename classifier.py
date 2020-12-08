from sqlalchemy.engine.url import URL
import sqlTemplates as sql
from jinja2 import Template
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text


class SaneProbabilityEstimator:

    def __init__(self, table_train, target=None, model_id='table_name', conn=None):
        """
        This method takes the most commonly used parameters from the user during initialization of the classifier
        - conn = database connection (format: variable)
        - target = target variable (what you are wanting to predict) (format: str)
        - table_name = name of the table that is in the database (format: str)
        """

        # Default: Embedded sqlite3 Database
        if conn is None:
            # Step 1.) Create new database
            sqlDB = sl.connect('SANE.db')
            # Step 2.) Establish path to new database
            self.conn = 'sqlite:///SANE.db'
            # Step 3.) Create engine to newly connected database
            self.engine = self.set_connection(self.conn)
            # Step 4.) Create a table in the DB - the dataframe the user will pass in
            table_train.to_sql(name='SANE_TABLE', con=sqlDB, if_exists='replace', index=False)
            self.table_train = 'SANE_TABLE'
        else:
            self.engine = self.set_connection(conn)
            self.table_train = table_train

        self.model_id = model_id
        self.target = target


    def set_connection(self, db):
        """
        :param db: dict with connection information for DB
        :return: SQLAlchemy Engine
        """
        if self.conn:
            return create_engine(self.conn)
        else:
            return create_engine(URL(**db))

    def get_connection(self):
        """
        :return: Connection to self.engine
        """
        return self.engine.connect()

    def execute(self, desc, query):
        connection = self.get_connection()
        print(desc + '\nQuery: ' + query)
        connection.execute(text(query))
        connection.close()
        print('OK: ' + desc)
        print()

    def executeQuery(self, desc, query):
        connection = self.get_connection()
        print('Query: ' + query)
        results = connection.execute(text(query))
        results = results.fetchall()
        connection.close()
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


    def train_test_split(self, seed=1, ratio=0.8):
        """
        Splitting table into training set and evaluation set
        :return: table_eval
        """

        self.seed = seed
        self.ratio = ratio

        self.materializedView(
             'Splitting table into training set',
             self.model_id + '_train',
             Template(sql.tmplt['_train']).render(input=self))

        self.materializedView(
             'Splitting table into test set',
             self.model_id + '_table_eval',
             Template(sql.tmplt['_table_eval']).render(input=self))


    def train(self, table_train, catFeatures, bins=50, numFeatures=None):
        """
        1.) Need to be feeding the train set (0.8) of original table into this --> training phase
        2.) Then, invoking the predict method on the test set (0.2) of the original table --> prediction phase on new data
        
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

        if numFeatures is None:
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

