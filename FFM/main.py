import data_loader
from FFM import FFM 

if __name__ == '__main__':
    path = '../data/data.csv'
    nrows = None
    feature_size, field_dict, data = data_loader.data_load('../data/data.csv',nrows=nrows)
    
    features = ['userId', 'movieId', 'tag']

    num = data.shape[0] * 4 // 5

    model = FFM(features, feature_size, field_dict, embedding_size=4)

    X = data[features].values
    y = data.label.values.reshape(-1,1)
    model.fit(
        X[:num],y[:num], epoch=10,
        X_valid=X[num:],y_valid=y[num:],
        early_stopping=True, refit=True
    )
