import pandas as pd 
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
import gc

sc = SparkContext()
sqlContext = SQLContext(sc) 
df = pd.read_csv('ratings.csv', nrows=None) 
movie_cate = pd.read_csv('movies.csv')[['movieId', 'genres']]

def helper(x):
    user_id = x[0]
    record = x[1]

    record = record.split('|')

    total_cnt = len(record)

    trian_cnt = int(total_cnt * 0.75)

    record = [item.split(',') for item in record]

    record = [
        [str(item[0]), float(item[1]), int(item[2])] for item in record
    ]

    movie_list = []

    record = sorted(record, key=lambda x: x[2])

    i = 0
    for item in record:
        label = int(item[1] >= 4)
        s = '|'.join(movie_list[-10:])
        yield user_id, item[0], s, label
        i += 1
        if label == 1 and i < trian_cnt:
            movie_list.append(item[0])

def helper_to_string(x):
    user_id = x[0]
    record = ','.join(map(str, x[1:]))
    yield user_id, record

for user in range(1, 30000):
    temp = df[df.userId == user]
    count = temp.shape[0]
    trian_cnt = int(count * 0.75)
    sdf = sqlContext.createDataFrame(temp)
    sdf = sdf.rdd.flatMap(helper_to_string).reduceByKey(lambda x,y: x+'|'+y).toDF(['user_id', 'record'])
    sdf = sdf.rdd.flatMap(helper).toDF(['user_id', 'movie_id', 'pos_movie_list', 'label'])
    temp = sdf.toPandas()
    temp[:trian_cnt].to_csv('train.csv',index=False,mode='a',header=user==1)
    temp[trian_cnt:].to_csv('test.csv',index=False,mode='a',header=user==1)
    del temp, sdf
    gc.collect()
    print ('finish users %6d , %4d samples~' % (user, count))









