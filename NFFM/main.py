# encoding = utf8
import data_loader
from NFFM import NFFM 
import sys
nrows = None 
if len(sys.argv) > 1:
    nrows = sys.argv[1]
    nrows = int(nrows)

if __name__ == '__main__':
    path = '../data/data.csv'

    feature_size, feature_field, data = data_loader.data_load('../data/data.csv', nrows=nrows)
    features = ['userId', 'movieId', 'tag']

    num = data.shape[0] * 4 // 5

    model = NFFM(features, feature_size, feature_field, embedding_size=4, layers=[200,200,200])

    X = data[features].values
    y = data.label.values.reshape(-1,1)
    model.fit(
        X[:num],y[:num], epoch=10,
        X_valid=X[num:],y_valid=y[num:],
        early_stopping=True, refit=True
    )
