import os
import pickle
import pandas as pd 

category_feature = ['user_id', 'movie_id']

multi_feature = ['pos_movie_list', 'genres']

features = category_feature + multi_feature

user_col = 'user_id'

item_col = 'movie_id'

history_col = 'pos_movie_list'

train_path = '../ratings/train.csv'

test_path = '../ratings/test.csv'

tags_path = '../ratings/tags.pkl'

max_len = 10


def build_feature_tags():
    df = pd.read_csv(train_path)
    feature_tags = {}
    for s in features:
        if s == history_col:
            continue
        df[s] = df[s].apply(str).fillna('')
        if s in category_feature:
            vals = set()
            for val in df[s]:
                vals.add(val)
        else:
            vals = set()
            for line in df[s]:
                vals = vals | set(line.strip('|').split('|'))
        feature_tags[s] = ['0'] + list(vals)
        print ('build %s tags done!' %s)
    feature_tags[history_col] = feature_tags[item_col]
    return feature_tags


if not os.path.exists(tags_path):
    feature_tags = build_feature_tags()
    item2index = {}
    for i in range(len(feature_tags[item_col])):
        item = feature_tags[item_col][i]
        item2index[item] = i
    pickle.dump(feature_tags, open(tags_path, 'wb'))
    pickle.dump(item2index, open('../ratings/item2index.pkl', 'wb'))
else:
    feature_tags = pickle.load(open(tags_path, 'rb'))
    item2index = pickle.load(open('../ratings/item2index.pkl', 'rb'))

