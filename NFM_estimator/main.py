# coding:utf-8
import logging
import os
import pickle
import tensorflow as tf
from args import args
from model import creat_graph
from data_loader import data_load, read_csv_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

# logging.basicConfig(level=logging.INFO,
# format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')


def get_estimator():
    checkpoint_dir = os.path.join(args.model_path, 'checkpoints')
    if args.last_model_path and not tf.train.latest_checkpoint(checkpoint_dir):
        warmup_dir = os.path.join(args.last_model_path, 'checkpoints')
    else:
        warmup_dir = None

    params = {
        'dropout': args.dropout,
        'embedding_size': args.embedding_size,
        'layers': args.layers,
        'learning_rate': args.learning_rate,
        'activation': args.activation,
        'feature_size': args.feature_size,
        'batch_norm': args.batch_norm
    }

    return tf.estimator.Estimator(
        model_fn=creat_graph,
        config=tf.estimator.RunConfig(
            keep_checkpoint_max=5,
            model_dir=checkpoint_dir,
            save_checkpoints_secs=1200,
            log_step_count_steps=300,
        ),
        params=params,
        warm_start_from=warmup_dir
    )


def main():
    df = data_load(args.file_path, args.nrows)
    num = df.shape[0] * 4 // 5
    train_df = df[:num]
    valid_df = df[num:]
    estimator = get_estimator()
    for _ in range(args.epochs):
        logging.info('==== Start to train ===>')
        estimator.train(
            input_fn=lambda: read_csv_data(train_df.values)
        )
        logging.info('==== Start to evaluate ===>')
        result = estimator.pridict(
            input_fn=lambda: read_csv_data(valid_df.values)
        )
        val_result = np.array([item['y_pre'] for item in result])
        logloss = log_loss(valid_df.label, val_result)
        auc = roc_auc_score(valid_df.label, val_result)
        print ('val auc = %f log loss = %f' %(auc, logloss))


if __name__ == '__main__':
    main()
