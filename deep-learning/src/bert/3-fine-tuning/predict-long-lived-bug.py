import keras.backend as K
from sklearn.model_selection import train_test_split
from official.nlp import optimization
import tensorflow_text as text
import tensorflow_hub as hub
import sys
import os
import logging
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.random.uniform([RANDOM_SEED], seed=RANDOM_SEED)


sys.path.append('data/models')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

logging.debug("Tf Version : %s", tf.__version__)
logging.debug("Eager mode : %s", tf.executing_eagerly())
logging.debug("Hub Version: %s", hub.__version__)
tf.get_logger().setLevel('ERROR')

LABELS = [0, 1]
MAX_LEN = 128
PROJECT = 'mozilla'

device_name = tf.test.gpu_device_name()

logging.debug("Reading traning and testing datasets")
raw_train_data = pd.read_csv(
    f'data/clean/20210305_{PROJECT}_bug_report_train_data.csv')

short, long = np.bincount(raw_train_data['label'])
total = short + long
logging.debug('Examples:\n Total: {} Long-lived bugs: {} ({:.2f}% of total) '.format(
    total, long, 100 * long / total))

raw_test_data = pd.read_csv(
    f'data/clean/20210305_{PROJECT}_bug_report_test_data.csv')

train_data, val_data = train_test_split(
    raw_train_data, test_size=.10, stratify=raw_train_data['label'], random_state=RANDOM_SEED)

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_data['long_description'].values, train_data['label'].values))
train_ds = train_ds.shuffle(buffer_size=1021).batch(64)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_data['long_description'].values, val_data['label'].values))
val_ds = val_ds.batch(64)

test_ds = tf.data.Dataset.from_tensor_slices(
    (raw_test_data['long_description'].values, raw_test_data['label'].values))
test_ds = test_ds.shuffle(buffer_size=1021).batch(64)

logging.debug("Downloading BERT preprocess handler and encoder")

BERT_MODELS = [
    "albert_en_base",
    "distilbert_en_uncased",
    "electra_small",
    "electra_base",
    "small_bert/bert_en_uncased_L-2_H-128_A-2",     # bert-tiny
    "small_bert/bert_en_uncased_L-4_H-256_A-4",     # bert-mini
    "small_bert/bert_en_uncased_L-4_H-512_A-8",     # bert-small
    "small_bert/bert_en_uncased_L-8_H-512_A-8",     # bert-medium
    "small_bert/bert_en_uncased_L-12_H-768_A-12"    # bert-base
]


def pick_bert_model(name='small_bert/bert_en_uncased_L-4_H-512_A-8'):
    """Returns the BERT model and preprocess handler based on name"""

    map_name_to_handle = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':  # (bert_mini)
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':  # (bert-small)
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':  # (bert-medium)
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':  # (bert-base)
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_base/3',
        'distilbert_en_uncased':
            'https://tfhub.dev/jeongukjae/distilbert_en_uncased_L-6_H-768_A-12/1',
        'electra_small':
            'https://tfhub.dev/google/electra_small/2',
        'electra_base':
            'https://tfhub.dev/google/electra_base/2',
        'experts_pubmed':
            'https://tfhub.dev/google/experts/bert/pubmed/2',
        'experts_wiki_books':
            'https://tfhub.dev/google/experts/bert/wiki_books/2',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
    }

    map_model_to_preprocess = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
        'distilbert_en_uncased':
            'https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/2',
        'electra_small':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'electra_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'experts_pubmed':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'experts_wiki_books':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    }

    handle_encoder = map_name_to_handle[name]
    handle_preprocess = map_model_to_preprocess[name]

    return handle_encoder, handle_preprocess

#from utils import check_binary


def compute_binary_specificity(y_true, y_pred):
    """Compute the confusion matrix for a set of predictions

    Parameters
    ----------
    y_pred  :   predicted values for a batch if samples (must be binary: 0 or 1)
    y_true  :   correct values for the set of samples used (must binary: 0 or 1)

    Returns
    -------
    out     :   the sensitivity
    """

     
    neg_y_true = tf.dtypes.cast(1 - y_true, dtype=tf.float32)
    neg_y_pred = tf.dtypes.cast(1 - y_pred, dtype=tf.float32)
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    

    #tn = tf.dtypes.cast(K.sum(tf.dtypes.cast(tf.math.logical_and(y_true == 0, y_pred == 0), dtype=tf.int16), axis=[-1]), dtype=tf.float32)
    #fp = tf.dtypes.cast(K.sum(tf.dtypes.cast(tf.math.logical_and(y_true == 0, y_pred == 1), dtype=tf.int16), axis=[-1]), dtype=tf.float32)

    result = tn / (tn + fp + K.epsilon())

    return result

def sensitivity_loss_wrapper():
    """A wrapper to create and return a function which computes the 
       sensivity loss
    """
    # define the function for your loss
    def sensitivity_loss(y_true, y_pred):
        return 1 - compute_binary_specificity(y_true, y_pred)

    return sensitivity_loss


def make_model(encoder_name, preprocess_name, metrics=None, epochs=10):
    """build classifier model"""

    if metrics is None:
        metrics = tf.keras.metrics.BinaryAccuracy(name='accuracy')

    # setup the BERT pre-trained model
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(preprocess_name, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(encoder_name, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    bert = outputs['pooled_output']

    # make fine-tuning over the pre-trained model
    task = tf.keras.layers.Dropout(0.1)(bert)
    task = tf.keras.layers.Dense(
        1, activation='sigmoid', name='classifier')(task)
    model = tf.keras.Model(text_input, task)

    # setup hyperparameters for model
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_spec = sensitivity_loss_wrapper()
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    init_lr = 3e-5

    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    model.compile(optimizer=optimizer,
                  loss=loss_spec,
                  metrics=metrics)
    return model


epochs = 30
layers, hidden, attention = 0, 0, 0
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]
today = datetime.date.today()
for bert_model in BERT_MODELS:
    logging.debug(f"PROJECT: {PROJECT} - MODEL: {bert_model}")
    encoder_name, preprocess_name = pick_bert_model(bert_model)

    classifier_model = make_model(
        encoder_name, preprocess_name, metrics=METRICS, epochs=epochs)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3)
    history = classifier_model.fit(
        x=train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callback])
    results = classifier_model.evaluate(test_ds)

    metrics = []
    for name, value in zip(classifier_model.metrics_names, results):
        metrics.append(
            [today, PROJECT, bert_model, layers, hidden, attention, len(raw_train_data), len(raw_test_data), epochs,
             name, value])

    metrics_df = pd.DataFrame(metrics, columns=['date', 'project', 'model', 'L#', 'H#', 'A#', 'train_rows', 'test_rows',
                                                'epochs', 'metric', 'value'])

    file_name = f"data/metrics/fine-tuning/{today.strftime('%Y%m%d')}_{PROJECT}_test_metrics.csv"
    if os.path.exists(file_name):
        metrics_prev_df = pd.read_csv(file_name)
        metrics_df = pd.concat([metrics_prev_df, metrics_df])

    metrics_df.to_csv(
        f"data/metrics/fine-tuning/{today.strftime('%Y%m%d')}_{PROJECT}_test_metrics.csv", index=False)
