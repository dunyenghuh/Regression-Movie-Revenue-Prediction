import pandas as pd
import numpy as np
from tabulate import tabulate
import tensorflow as tf
from datetime import datetime


# https://towardsdatascience.com/natural-language-processing-with-tensorflow-e0a701ef5cef
# https://medium.com/towards-artificial-intelligence/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0


test0 = pd.read_csv('data/test.csv')
train0 = pd.read_csv('data/train.csv')

# impute runtime with the mean value. we are missing 2 values
# train0['runtime'] = train0.runtime.fillna(train0.runtime.mean())

# runtime for Королёв: 130  (https://www.imdb.com/title/tt1107828/)
# runtime for Happy Weekend: 90  (https://de.wikipedia.org/wiki/Happy_Weekend_(1996))
train0.loc[(train0['title'] == 'Королёв') & (train0['release_date'] == '10/29/07'), 'runtime'] = 130
train0.loc[(train0['title'] == 'Happy Weekend') & (train0['release_date'] == '3/14/96'), 'runtime'] = 130
train0.runtime.describe()

# remove rumored films
print(train0['status'].value_counts())
train1 = train0[train0['status'] != 'Rumored']

# remove unnecessary cols
remove = ['id', 'imdb_id', 'homepage', 'poster_path', 'spoken_languages', 'original_title', 'status']
train1 = train1.drop(columns=remove)

# get date in the right format
ymdcol = train1['release_date'].str.split("/", expand=True)
ymdcol.columns = ['month', 'day', 'year']
ymdcol['month'] = ymdcol['month'].astype(int)
ymdcol['day'] = ymdcol['day'].astype(int)
ymdcol['year'] = ymdcol['year'].astype(int)
ymdcol['year1'] = np.where(ymdcol['year'] > 17, 1900 + ymdcol['year'], 2000 + ymdcol['year'])
ymdcol['ymd'] = ymdcol['year1'].astype(str) + '-' + ymdcol['month'].astype(str) + '-' + ymdcol['day'].astype(str)
ymdcol['ymd'] = pd.to_datetime(ymdcol['ymd'])
ymdcol['ymd'] = ymdcol['ymd'].dt.date

train1['ymd'] = ymdcol['ymd']
train1['year'] = ymdcol['year1']
train1['month'] = ymdcol['month']
train1['day'] = ymdcol['day']

# json data
