# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE (vs. MADlib)
# (c) 2021        Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

# coding: utf-8
from SaneLib import SaneLib
from DataBase import MetaData
import constants as cns
import time
import matplotlib.pyplot as plt

# starting time
start = time.time()

sl = SaneLib({
    'drivername': 'mysql+mysqlconnector',
    'host': 'dbs-mysql.el.eee.intern',
    'port': 3306,
    'username': 'user',
    'password': 'jf84£DppDP_$ie94',
    'database': 'user',
    'query': {'charset': 'utf8'}
})
# '204673297'



#sl.db.createView('steam_games_subsample_train', 'select * from steam.steam_games  limit 100000', materialized=True)
#sl.db.createView('steam_games_subsample_eval', 'select * from steam.steam_games   limit 25000 offset 100000', materialized=True)

# sl.db.createView('steam_games3_train', 'select * from steam_games3  limit 150000000', materialized=False)
# sl.db.createView('steam_games3_eval', 'select * from steam_games3   limit 54673297 offset 150000000', materialized=False)

model_id='steam_subsample'

table_train='steam_games_subsample_train'

cat_cols = [
    'user_personastate_c',
    'user_communityvisibilitystate_c',
    'user_profilestate_c',
    'user_commentpermission_c',
    'user_primaryclanid_c',
    'user_loccountrycode_c',
    'game_Genre__maj_c',
    'game_Genre__min_c',
    'game_Developer_maj_c',
    'game_Developer_min_c',
    'game_publisher_maj_c',
    'game_publisher_min_c'
]

num_cols = [
    'user_realnamegiven_b',
    'user_lastlogoff_n',
    'game_resease_date_n',
    'game_count_distinct_achievements_n',
    'game_sum_achievement_percentage_n',
    'game_avg_achievement_percentage_n',
    'game_min_achievement_percentage_n',
    'game_max_achievement_percentage_n',
    'game_stddev_achievement_percentage_n'
]

bins = 15

target = 'target_playtime_class_n'

mdh = sl.mdh(model_id)

mdh.descriptive_statistics(table_train, cat_cols,num_cols, bins)

mdh.contingency_table_1d(table_train, cat_cols,num_cols, bins, target)

ranked_columns = mdh.rank_columns_1d()

# print(ranked[ranked["mi"]>= 0.1 ].head(7))

print(f"-- Runtime of the program is {time.time() - start} seconds")

# Training phase: _qt is trained on 0.8 of table ; _qmt based off of _qt ; _m based off of _qt

mdh.train(
    table_train='steam_games_subsample_train',
    catFeatures=['game_Developer_min_c', 'game_publisher_min_c'],

    numFeatures=['game_sum_achievement_percentage_n','game_resease_date_n', 'user_lastlogoff_n'],
    target=target,
    bins=35
)

print(f"-- Runtime of the program is {time.time() - start} seconds")

mdh.predict(table_eval='steam_games_subsample_eval')
print(f"-- Runtime of the program is {time.time() - start} seconds")

mdh.accuracy()

mdh.update_bayes()
print(f"-- Runtime of the program is {time.time() - start} seconds")

mdh.accuracy()
