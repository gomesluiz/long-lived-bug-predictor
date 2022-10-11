import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers as ppb

#PROJECTS   = ['eclipse', 'freedesktop', 'gnome', 'gcc', 'mozilla', 'winehq']
PROJECTS = ['winehq']
INPUT_DATA_PATH = 'data/clean/'
OUTPUT_DATA_PATH = 'data/tensors/'
OUTPUT_LOGS_PATH = 'output/logs/'
BATCH_SIZE = 100
DEBUG_ON = 1
REFERENCE = '20210305'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_tensors(descriptions, tokenizer, max_tokens=128):
    # tokenization.
    sentences = descriptions[1].apply(
        (lambda s: ' '.join(s.split()[:max_tokens])))
    tokenized = sentences.apply(
        (lambda s: tokenizer.encode(s, add_special_tokens=True, truncation=True)))

    # padding
    max_len = max_tokens
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # masking
    attention_mask = np.where(padded != 0, 1, 0)

    # model#1
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    return (input_ids, attention_mask)


def extract_features(dataset, model, tokenizer):
    
    bug_ids = dataset[0]

    input_ids, attention_mask = build_tensors(dataset, tokenizer)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    
    labels  = dataset[2]
    
    return (features, labels, bug_ids)


if DEBUG_ON:
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
else:
    logging.basicConfig(filename=os.path.join(OUTPUT_LOGS_PATH, 'extract-feature-w-bert.log'),
                        filemode='w', format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

logging.info('Torch version: {}'.format(torch.__version__))
logging.info('Transformers version: {}'.format(ppb.__version__))
logging.info('Importing BERT base uncased model')
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel,
                                                    ppb.BertTokenizer, 'bert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
model.to(device)


def read_reports_data(filename, batch_size=BATCH_SIZE):
    filepath = os.path.join(INPUT_DATA_PATH, filename)
    reader = pd.read_csv(filepath, header=None, skiprows=1,
                         chunksize=batch_size, low_memory=False)
    return reader


for project in PROJECTS:
    logging.info('Starting processing {} dataset'.format(project))

    logging.info('Loading training set')
    train_file_reader = read_reports_data(
        '{}_{}_bug_report_train_data.csv'.format(REFERENCE, project), BATCH_SIZE)

    for i, train_batch in enumerate(train_file_reader):
        logging.info('Extracting features from training batch {}'.format(i+1))
        train_features, train_labels, train_bug_ids = extract_features(
            train_batch, model, tokenizer)
        logging.info('Training features and labels shapes: {}, {}, {}'.format(
            train_features.shape, train_labels.shape, train_bug_ids.shape))

        logging.info('Saving training features')
        tensors_output_path = os.path.join(
            OUTPUT_DATA_PATH, '{}_{}_bug_report_train_data_{}.pt'.format(REFERENCE, project, i+1))
        torch.save(np.column_stack((train_bug_ids, train_features, train_labels)),
                   tensors_output_path)

    logging.info('Loading testing set')
    test_file_reader = read_reports_data(
        '{}_{}_bug_report_test_data.csv'.format(REFERENCE, project), BATCH_SIZE)

    for i, test_batch in enumerate(test_file_reader):
        logging.info('Extracting features from testing batch {}'.format(i+1))
        test_features, test_labels, test_bug_ids = extract_features(
            test_batch, model, tokenizer)
        logging.info('Testing features and labels shapes : {}, {}'.format(
            test_features.shape, test_labels.shape))

        logging.info('Saving testing features')
        tensors_output_path = os.path.join(
            OUTPUT_DATA_PATH, '{}_{}_bug_report_test_data_{}.pt'.format(REFERENCE, project, i+1))
        torch.save(np.column_stack((test_bug_ids, test_features, test_labels)),
                   tensors_output_path)

    logging.info('Processing {} dataset finished.'.format(project))

logging.info('Finishing extraction')
