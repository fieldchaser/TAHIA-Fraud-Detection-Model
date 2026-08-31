DATADIR = 'datasets'

num_class = 2

WEIBO = 'weibo'
REDDIT = 'reddit'
TOLOKERS = 'tolokers'
AMAZON = 'amazon'
TFINANCE = 'tfinance'
YELP = 'yelp'
QUESTIONS = 'questions'
DGRAPHFIN = 'dgraphfin'
TSOCIAL = 'tsocial'

TRAIN = '_train.txt'
VAL = '_val.txt'
TEST = '_test.txt'

import random
random.seed(1)
PARAMETERS = {WEIBO: [0.001, 6, 0, 0.5, 1, random.uniform(0.01,0.02), random.uniform(0.01,0.02)],
              REDDIT: [0.001, 5, 0.05, 0, 0, 0, random.uniform(0.01,0.02)],
              TOLOKERS: [0.001, 1, 0.05, 0, 0.5, 0, random.uniform(0.01,0.02)],
              AMAZON: [0.001, 3, 0.05, 0, 0, 0, random.uniform(0.01,0.02)],
              TFINANCE: [0.001, 3, 0.1, 1, 1, random.uniform(0.01,0.02), random.uniform(0.01,0.02)],
              YELP: [0.0001, 1, 0.1, 0, 0, 0, random.uniform(0.01,0.02)],
              QUESTIONS: [0.0001, 6, 0.1, 0.5, 0.5, 0, 0],
              DGRAPHFIN: [0.001, 4, 0.05, 0, 1, random.uniform(0.01,0.02), random.uniform(0.01,0.02)],
              TSOCIAL: [0.001, 6, 0.05, 0.5, 1, random.uniform(0.01,0.02), random.uniform(0.01,0.02)]}

def set_paras(data):
    return PARAMETERS[data]
