# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2020  Michael Kaufmann, Anna Huber, Gabriel Stechschulte, HSLU

# coding: utf-8

import mysql.connector as mysql

db = mysql.connect(
    host="localhost",
    user="root",
    passwd="",
    database='dbml')

class SaneProbabilityEstimator:

    def __init__(self, conn, table_train, target, model_id):
        '''
        This method takes the most commonly used parameters from the user during initialization of the classifier
        - conn = database connection (format: variable)
        - target = target variable (what you are wanting to predict) (format: str)
        - table_name = name of the table that is in the database (format: str)
        '''
        self.connection = conn
        self.table_train = table_train
        self.target = target # TODO default value: last column
        self.model_id = model_id # TODO default value: table name

# TODO use SQL alchemy or similar framework that works for all SQL DB types
    def execute(self, desc, query):
        cursor = self.connection.cursor()
        print('Query: ' + query)
        cursor.execute(query)
        cursor.close()
        print('OK: ' + desc)
        print()

    def materializedView(self, desc, tablename, query):
        self.execute('Dropping table ' + tablename, '''
            drop table if exists {}'''
                .format(tablename))
        self.execute(desc, '''
            create table {} as'''
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
    def hyperparameters(self, features, bins):
        '''
        This function sets the hyperparameters
        - features to estimate the probability of the target
        - features = right now it is just a string, but I think we may want to explore
                     having the user put the feature(s) into a numpy array. Similiar to
                     how the scikit-learn ML algorithms want them. (format:str)
        - Bins/buckets = # of bins
        '''
        # TODO array of features
        self.features = features # TODO default: all columns
        # TODO array of bins
        self.bins = bins; #TODO default: 10

    def train(self):
        '''
        This function is the training phase:
        - the input data table is quantized (equal size) and indexed.
        - This quantized index then represents an in-database model for probability estimation
        '''
        # make sure only 1 query is executed per call. So it works in PyCharm.

        # TODO Generate queries using n features x1, x2, ..., xn; differentiate between numerical and categorical

        self.materializedView('''Quantization of training table''', self.model_id + '''_qt''', '''
            select y, 
            xq1,  min(xn1) as mn1, max(xn1) as mx1, 
            count(*) as n
             from ( 
                select
                    CEIL({}*RANK() OVER (ORDER BY {})*1.0/COUNT(*) OVER()) as xq1,
                    {} as xn1,
                    {} as y
                from {} )a
            group by y, xq1;'''
                .format(
                    self.bins, self.features,
                    self.features,
                    self.target,
                    self.table_train))

        self.materializedView('Quantization metadata for training table', self.model_id + '_qmt', '''
            select 
                a.* ,
                min(mn) over(partition by i) as tmn,
                max(mx) over(partition by i) as tmx
            from (
                select 1 as i, xq1 as xq, min(mn1) as mn,  max(mx1) as mx from {}_qt group by xq1
            ) a;'''
                .format(
                    self.model_id))
        self.materializedView('Computing contingency table ', self.model_id + '_m', '''
            select
            y as y_,
            xq1, mn1, mx1, mx_1, tmn1, tmx1, # num. attr.
            n__, n_y,
            sum(n) as nxy
            from (
            select
                y,
                xq1, # num. attr. 1
                x1.mn as mn1,
                x1.mx as mx1,
                x1.mx_ as mx_1,
                min(mn1) OVER( ) as tmn1,
                max(mx1) OVER( ) as tmx1,
                sum(n) over() as n__,
                sum(n) over(partition by y) as n_y,
                n
            from {}_qt as t
            join (select i, xq, mn, mx, LEAD(mn, 1) OVER (ORDER BY xq) as mx_ from {}_qmt where i = 1) x1 on (xq1 = x1.xq)
            ) a
            group by y,
            xq1, mn1, mx1, mx_1, tmn1, tmx1, # num. attr.
            n__, n_y;'''
                .format(
                    self.model_id,
                    self.model_id))


    def predict(self, table_eval):
        '''
        This function estimates the probabilities for the evaluation data
        '''

        self.materializedView('Quantization metadata for training table', self.model_id + '_qe', '''
            select t.*, x1.xq as xq1
                from 
                (select row_number() over() as id, {} as y, {} as xn1
             from {})
                as t
                join (select i, xq, mn, mx, tmn, tmx, LEAD(mn, 1) OVER (ORDER BY xq) as mx_ from {}_qmt where i = 1) x1 
                    on ((x1.mx_ > xn1 or x1.mx = x1.tmx) and (x1.mn <= xn1 or x1.mn = x1.tmn)) # num. attr. 1
            ;'''
                .format(
                    self.target,
                    self.features,
                    table_eval,
                    self.model_id))

        self.materializedView('Class prediction for evaluation dataset', self.model_id + '_p', '''
            select *
            from ( -- b
                select id, xn1, y, y_,
                    case when max(p_xgy * p_y) OVER(PARTITION BY xq1) -- arg max y (p_xgy * p_y)
                        = p_xgy * p_y then 1 else 0 end as prediction
                from --
                (
                    select
                        e.*, m.y_,
                        case when m.nxy is null then 1.0 else
                        m.nxy * 1.0 / m.n_y 
                        end  -- default predictor
                        as p_xgy, -- sampled probability of x in [mn, mx] given y
                         (m.n_y  * 1.0 / m.n__) as p_y -- probability of y
                    from {}_qe as e -- evaluation  dataset
                    left outer join {}_m as m
                    using(xq1)
                ) a
            ) b
            where prediction = 1 ;'''.format(
                    self.model_id,
                    self.model_id))

    def accuracy(self):
        '''
        Computing the accuracy of the model and returning the results to the user
        '''

        # todo def execute Query that returns results
        cursor5 = self.connection.cursor()
        query = '''
            select
            count(distinct id) as  cases,
            sum(case when e.y = p.y_ then 1 else 0 end) as tp,
            sum(case when e.y = p.y_ then 1 else 0 end) /  count(distinct id)  as accuracy
            from {}_qe e 
            left outer join {}_p p using(id);
                '''.format(
                    self.model_id,
                    self.model_id)

        cursor5.execute(query)
        results = cursor5.fetchall()
        cursor5.close()

        accuracy = [result[2] for result in results]
        print('Model accuracy = ', accuracy)

## Other userful functions / methods for data scientists:
## * Get list of features ranked by 1D prediction accuracy
## * Optimize n Buckets for 1 feature
## * Visualize 1D and 2D Decision Table
## * Load Data into a DB Table
## * Split Training / Test
## * Remove Model from DB (drop all tables for model_id)


# **I plan on making this more functional**: Make it so not as many functions need to be called

# Initialization with the required variables
# As stated in the documentation above, I think we may want to have the user pass in the target and feature
# variables similiar to how it is done in skikit - I.e: Numpy arrays, etc. But not sure how SQL would like this
# MK --> perhaps SQLalchemy can be of help here.
classifier = SaneProbabilityEstimator(db, 'covtypall', 'Cover_Type', 'covtyptest')

classifier.hyperparameters('Elevation', 44)

# There may be a better way in which the user passes parameters to this function - Searching for solutions
# The SaneProbabilityEstimator Object holds all parameters
## perhaps pass the hyperparaemters directly to train method
classifier.train()

# Would like to have this function be more like the scikit framework - I.e: classifer.predict(y_test) - Searching for solutions
#Input: name of table with same structure as table_train
classifier.predict('table_eval')

# The print statement of this function needs cleaned up - Working on it
classifier.accuracy()
