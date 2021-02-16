import Analysis
from jinja2 import Template
import sqlTemplates as sql
import matplotlib.pyplot as plt
import pandas as pd
import Utils
from plotnine import ggplot, aes, geom_line, geom_point, labs, theme, element_text
from plotnine.data import economics, mpg
import plotnine as pn
import Prediction

class Model():
    def __init__(self,analysis_obj,bins=50):
        if analysis_obj != None:
            self.analysis = analysis_obj
            self.cat_feat = self.analysis.get_cat_feat()
            self.num_feat = self.analysis.get_num_feat()

        self.utils = Utils()
        database = self.analysis.database

        self.bins = bins


    def visualize1D(self, feature1):

        numFeats = self.numFeatures

        if feature1 in numFeats:
            index = self.numFeatures.index(feature1) + 1
        else:
            raise ValueError('Feature variable {} does not exist'.format(feature1))

        for i in range(1, len(numFeats) + 1):
            if index == i:
                feature1 = 'xn{}'.format(i)
                bins = 'xq{}'.format(i)
                minimum = 'mn{}'.format(i)
                maximum = 'mx_{}'.format(i)

        hist = self.executeQuery('Discretization Histogram', '''
            select distinct {} as xq, {} as mn, {} as mx, 
            concat({}, ': ]', {}, ',', {}, ']') as bin, 
            cast(y_ as char) as y_,
            sum(nxy)over(partition by {}, y_)*1.0/
            sum(nxy) over(partition by y_)  as p 
            from {}_m 
            order by {}, y_;'''.format(bins, minimum, maximum,
                                       bins, minimum, maximum,
                                       bins, self.model_id, bins))

        hist_df = pd.DataFrame(hist)
        hist_df.columns = ['xq', 'mn', 'mx', 'bin', 'y_', 'p']
        hist_df[['p']] = hist_df[['p']].apply(pd.to_numeric)
        t = hist_df[['bin']].drop_duplicates()

        bin_list = hist_df['bin'].drop_duplicates().tolist()
        bin_cat = pd.Categorical(hist_df['bin'], categories=bin_list)

        p = (
                ggplot(hist_df)
                + aes(bin_cat, 'p', color='y_', group='y_')
                + geom_point()
                + geom_line()
                + labs(y='p(qx|y)', x='bin', title='test')
                + theme(axis_text_x=element_text(rotation=90, hjust=1))
        )

        print(p)

    def predict(self):
        return Prediction.Prediction(self.analysis.eval,self.analysis.model_id)
