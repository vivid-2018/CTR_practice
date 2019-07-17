# coding = utf-8
from args import *
from dice import dice
from deepfm import DeepFM
import sys

nrows = None
if len(sys.argv) >= 2:
    nrows = int(sys.argv[1])


if __name__ == '__main__':
    train = pd.read_csv(train_path, nrows=None).fillna('0')
    test = pd.read_csv(test_path, nrows=None).fillna('0')
    for s in features:
        train[s] = train[s].apply(str)
        test[s] = test[s].apply(str)
    train_x, train_y = train[features].values, train['label'].values.reshape((-1,1))

    test_x, test_y = test[features].values, test['label'].values.reshape((-1,1))

    model = DeepFM(features, feature_tags, activate=dice)

    model.fit(train_x, train_y, X_valid=test_x, y_valid=test_y)

