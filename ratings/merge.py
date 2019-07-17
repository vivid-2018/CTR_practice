import pandas as pd 

train = pd.read_csv('train.csv')[['user_id', 'movie_id', 'pos_movie_list', 'label']]
test = pd.read_csv('test.csv')[['user_id', 'movie_id', 'pos_movie_list', 'label']]

genres = pd.read_csv('movies.csv')[['movieId', 'genres']]

train = train.merge(genres, left_on='movie_id', right_on='movieId', how='left')
train.drop(columns='movieId', inplace=True)
train.to_csv('train.csv',index=False)

test = test.merge(genres, left_on='movie_id', right_on='movieId', how='left')
test.drop(columns='movieId', inplace=True)
test.to_csv('test.csv',index=False)


