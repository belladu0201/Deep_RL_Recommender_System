import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from utility import to_pickled_df, pad_history

def parse_args():
    parser = argparse.ArgumentParser(description="Create replay_buffer dataframe and data_statis.df")

    parser.add_argument('--data', nargs='?', type = str,default='data',
                        help='data directory')
    parser.add_argument('--state_len', type = int, default = 10, help = 'Max state length')
    return parser.parse_args()

if __name__ == '__main__':
    data_directory = 'data'
    print('Generating replay_buffer from sorted_events')
    args = parse_args()
    length = args.state_len

    # reply_buffer = pd.DataFrame(columns=['state','action','reward','next_state','is_done'])
    sorted_events  = pd.read_pickle(os.path.join(data_directory, 'sorted_events.df'))
    item_ids = sorted_events.article_id.unique()
    pad_item = len(item_ids)

    train_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
    groups = train_sessions.groupby('customer_id')
    ids = train_sessions.customer_id.unique()
    print('There are {} ids to process'.format(len(ids)))
    state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [], [], []



    for i,id in zip(tqdm(range(len(ids))),ids):
        group = groups.get_group(id)
        history = []
        for index, row in group.iterrows():
            s = list(history)
            len_state.append(length if len(s) >= length else 1 if len(s) == 0 else len(s))
            s = pad_history(s, length, pad_item)
            a = row['article_id']
            is_b = row['is_buy']
            state.append(s)
            action.append(a)
            is_buy.append(is_b)
            history.append(row['article_id'])
            next_s = list(history)
            len_next_state.append(length if len(next_s) >= length else 1 if len(next_s) == 0 else len(next_s))
            next_s = pad_history(next_s, length, pad_item)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1] = True

    dic = {'state': state, 'len_state': len_state, 'action': action, 'is_buy': is_buy, 'next_state': next_state,
           'len_next_states': len_next_state,
           'is_done': is_done}
    reply_buffer = pd.DataFrame(data=dic)
    to_pickled_df(data_directory, replay_buffer=reply_buffer)

    print('Generated the replay_buffer \n')

    dic = {'state_size': [length], 'item_num': [pad_item]}
    data_statis = pd.DataFrame(data=dic)
    to_pickled_df(data_directory, data_statis=data_statis)
    print('Generated the data_statis.df \n')
    print("\nScript completed successfully!")