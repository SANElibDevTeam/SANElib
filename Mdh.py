
import sqlTemplates as sql
from jinja2 import Template
import pandas as pd
import time

from plotnine import ggplot, aes, geom_line, geom_point, scale_x_continuous, \
    scale_x_discrete, geom_col, theme_light, labs, theme, element_text, theme_bw, facet_wrap

from DataBase import DataBase

class Mdh:

    def __init__(self, sl, model_id):
        """
        This method takes the most commonly used parameters from the user during initialization of the classifier
        - conn = database connection (format: variable)
        - target = target variable (what you are wanting to predict) (format: str)
        - table_name = name of the table that is in the database (format: str)
        """
        self.sl = sl
        self.db = sl.db
        self.model_id = model_id

    def descriptive_statistics(self, table_train, catFeatures, numFeatures, bins):
        self.table_train = table_train
        self.catFeatures = catFeatures
        self.numFeatures = numFeatures
        self.bins = bins
        self.db.createView(self.model_id + '_agg', Template(sql.tmplt['_agg']).render(input=self),
                           desc='Get aggregation values for cols')

    def contingency_table_1d(self, table_train, catFeatures, numFeatures, bins, target):
        start = time.time()
        self.table_train = table_train
        self.catFeatures = catFeatures
        self.numFeatures = numFeatures
        self.bins = bins
        self.target = target
        self.db.execute('drop table if exists '+ self.model_id + '_m1d')
        self.db.execute('create table ' + self.model_id + '_m1d(f text, t text, x text, y_ text, nxy bigint)')
        for c in catFeatures:
            self.col = c
            self.db.execute(Template(sql.tmplt["_m1d_cat"]).render(input=self))
        for n in numFeatures:
            self.col = n
            self.db.execute(Template(sql.tmplt["_m1d_num"]).render(input=self))

    def rank_columns_1d(self):
        results = self.db.executeQuery(Template(sql.tmplt["_m1d_mi"]).render(input=self),
                                       'Computing mutual information with target')
        df = pd.DataFrame(results)
        df.columns = ['f', 't', 'mi']
        print(df)
        self.ranked_columns = df
        return df

    def train(self, table_train, catFeatures, numFeatures, target, bins):
        """

        """
        self.table_train = table_train
        self.catFeatures = catFeatures
        self.numFeatures = numFeatures
        self.target = target
        self.bins = bins

        #if not(self.db.does_table_exist(self.model_id + '_agg')):
        self.db.createView(self.model_id + '_agg', Template(sql.tmplt['_agg']).render(input=self),
                           desc='Get aggregation values')
        self.db.createView(self.model_id + '_qt', Template(sql.tmplt['_qt']).render(input=self),
                           desc='Quantization of training table')

    def predict(self, table_eval):  # table_eval is the test set
        """
        This function estimates the probabilities for the evaluation data
        """

        self.table_eval = table_eval

        self.db.createView(self.model_id + '_qe', Template(sql.tmplt['_qe']).render(input=self),
                           desc='Quantization metadata for evaluation table')  ## generate SQL using Jinja 2 template
        self.db.createView(self.model_id + '_p', Template(sql.tmplt['_p']).render(input=self),
                           desc='Class prediction for evaluation dataset')  ## generate SQL using Jinja 2 template

    def update_bayes(self, dim=1):
        self.db.execute(Template(sql.tmplt['_p_update']).render(input=self),
                        'Updating prediction with default prediction for null predictions')

    def accuracy(self):
        """
        Computing the accuracy of the model and returning the results to the user
        """
        results = self.db.executeQuery('''
            select
            count(distinct id) as  cases,
            sum(case when e.y = p.y_ then 1 else 0 end) as tp,
            sum(case when e.y = p.y_ then 1 else 0 end) /  count(distinct id)  as accuracy
            from {}_qe e 
            left outer join {}_p p using(id)
                '''.format(
            self.model_id,
            self.model_id), 'Computing evaluation accuracy')

        df = pd.DataFrame(results)
        df.columns = ['Total', 'TP', 'Accuracy']
        print(df)
