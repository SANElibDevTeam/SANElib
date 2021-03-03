from jinja2 import Template
import sqlTemplates as sql
import pandas as pd
import Database
from Utils import *
from sqlalchemy import text
import Model

class Analysis():
    def __init__(self,engine,dataset,target=None,seed=None,model_id='table_name',ratio=0.8):
        self.engine = engine
        self.dataset = dataset
        self.model_id = model_id
        self.seed = seed
        self.ratio = ratio
        self.train = dataset
        self.eval = self.train
        if ratio != 1.0:
            self.train_test_split()
            self.train = self.model_id + '_train'
            self.eval = self.model_id + '_table_eval'
        if target is None:
            col = self.executeQuery('If last column (DESC), else first column (ASC)', '''
                SELECT
                COLUMN_NAME
                ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{}'  #table_train
                ORDER BY ORDINAL_POSITION DESC
                LIMIT 1;
                '''.format(dataset))

            # ".fetchall" returns a list of RowProxys (SQLAlchemy) so this
            # for loop manually parses for the last column in table
            for index, element in enumerate(col):
                element = tuple(element)
                self.target = element[index]
                print(self.target)
        else:
            self.target = target


    def get_cat_feat(self):

        return self.catFeatures


    def get_num_feat(self):
        return self.numFeatures

    # TODO: Write get drop func
    def drop(self):
        pass

    def train_test_split(self):
        """
        Splitting table into training set and evaluation set
        Update eval
        """

        materializedView(
            'Splitting table into training set',
            self.model_id + '_train',
            Template(sql.tmplt['_train']).render(input=self), self.engine)

        materializedView(
            'Splitting table into test set',
            self.model_id + '_table_eval',
            Template(sql.tmplt['_table_eval']).render(input=self), self.engine)

    def rank(self, table_train, catFeatures, numFeatures, bins):
        self.numFeatures = numFeatures
        self.bins = bins
        self.catFeatures = catFeatures
        materializedView(
            'Computing 1d contingecies with target',
            self.model_id + '_m1d',
            Template(sql.tmplt['_m1d']).render(input=self),self.engine)
        results = executeQuery('Computing mutual information with target',
                                    Template(sql.tmplt["_m1d_mi"]).render(input=self),self.engine)
        df = pd.DataFrame(results)
        df.columns = ['f', 'mi']
        print(df)
        return self


    def estimate(self, catFeatures, bins=50, numFeatures=None):
        """
        1.) Need to be feeding the train set (ratio) of original table into this --> training phase
        2.) Then, invoking the predict method on the test set (1-ratio) of the original table --> prediction phase on new data

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
            allColumns = executeQuery('Querying for all of the columns', '''
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{}';  #table_train
            '''.format(self.train), self.engine)

            self.numFeatures = [element for index, tuple in enumerate(allColumns) for index, element in enumerate(tuple)]
        else:
            self.numFeatures = numFeatures

        self.bins = bins
        self.catFeatures = catFeatures

        # TODO Generate queries using n features x1, x2, ..., xn; differentiate between numerical and categorical

        materializedView(
            'Quantization of training table',
            self.model_id + '_qt',
            Template(sql.tmplt['_qt']).render(input=self), self.engine)
        materializedView(
            'Quantization metadata for training table',
            self.model_id + '_qmt',
            Template(sql.tmplt['_qmt']).render(input=self), self.engine)
        materializedView(
            'Computing predictive model as contingency table',
            self.model_id + '_m',
            Template(sql.tmplt['_m']).render(input=self), self.engine)

        return Model.Model(self)