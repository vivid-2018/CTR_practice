import data_loader
from FFM import FFM 
import sys
nrows = None 
if len(sys.argv) > 1:
    nrows = sys.argv[1]
    nrows = int(nrows)

if __name__ == '__main__':
    path = '../data/data.csv'
    feature_size, field_dict, data = data_loader.data_load('../data/data.csv',nrows=nrows)
    
    features = ['userId', 'movieId', 'tag']

    num = data.shape[0] * 4 // 5

    model = FFM(features, feature_size, field_dict, embedding_size=4, verbose=False)

    X = data[features].values
    y = data.label.values.reshape(-1, 1)
    '''
    model.fit(
        X[:num],y[:num], epoch=20,
        X_valid=X[num:],y_valid=y[num:],
        early_stopping=True, refit=True
    )
    '''
    import time

    start = time.time()
    model.fit(X[:num], y[:num], epoch=1)
    print('train a epoch cost %.2f' % (time.time() - start))

    start = time.time()
    model.predict(X[num:])
    print('predict cost %.2f' % (time.time() - start))

