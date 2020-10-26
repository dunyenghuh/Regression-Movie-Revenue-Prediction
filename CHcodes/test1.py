import pandas as pd
import numpy as np
from tabulate import tabulate
import tensorflow as tf
from datetime import datetime
import ast


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

# missing rows
print(train1.isna().sum())

# remove unnecessary cols
remove = ['id', 'imdb_id', 'homepage', 'poster_path', 'spoken_languages', 'original_title', 'status',
          'belongs_to_collection', 'tagline', 'Keywords']
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

# dict data


# genre
def dictcol(col0, col1):
    gen0 = []
    for g in train1[col0]:
        try:
            if pd.isnull(g):
                gen1 = []
            else:
                gen1 = [i[col1] for i in ast.literal_eval(g)]
            gen0.append(gen1)
        except ValueError:
            print(g)
    return gen0


train1['genre0'] = dictcol('genres', 'name')
train1['ngen'] = train1['genre0'].apply(lambda x: len(x))

train1['company'] = dictcol('production_companies', 'name')
train1['ncomp'] = train1['company'].apply(lambda x: len(x))

train1['country'] = dictcol('production_countries', 'name')
train1['ntry'] = train1['country'].apply(lambda x: len(x))
