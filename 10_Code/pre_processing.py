import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from sklearn.preprocessing import LabelEncoder
from utility import to_pickled_df, pad_history

def parse_args():
    parser = argparse.ArgumentParser(description="execute preprocessing steps")

    parser.add_argument('--data', nargs='?', type = str,default='data',
                        help='data directory')
    parser.add_argument('--state_len', type = int, default = 10, help = 'Max state length')
    return parser.parse_args()

if __name__ == '__main__':
    data_directory = 'data'
    print('Starting pre-processing of data')
    event_df = pd.read_csv(os.path.join(data_directory, 'transactions_train_small.csv'), header=0)
    event_df.columns = ['t_dat','customer_id','article_id','price','sales_channel_id']
    event_df = event_df.drop_duplicates()
    ##########remove users with <=2 interactions
    event_df['valid_session'] = event_df.customer_id.map(event_df.groupby('customer_id')['article_id'].size() > 2)
    event_df = event_df.loc[event_df.valid_session].drop('valid_session', axis=1)
    ##########remove items with <=2 interactions
    event_df['valid_item'] = event_df.article_id.map(event_df.groupby('article_id')['customer_id'].size() > 2)
    event_df = event_df.loc[event_df.valid_item].drop('valid_item', axis=1)
    ######## transform to ids
    article_encoder = LabelEncoder()
    customer_encoder= LabelEncoder()
    sales_encoder=LabelEncoder()
    event_df['article_id'] = article_encoder.fit_transform(event_df.article_id)
    event_df['customer_id'] = customer_encoder.fit_transform(event_df.customer_id)
    event_df['sales_channel_id']=sales_encoder.fit_transform(event_df.sales_channel_id)
    ###########sorted by user and timestamp
    event_df['is_buy'] = event_df['sales_channel_id']
    event_df = event_df.drop('sales_channel_id', axis=1)
    sorted_events = event_df.sort_values(by=['customer_id', 't_dat'])

    sorted_events.to_csv('data/sorted_events.csv', index=None, header=True)

    to_pickled_df(data_directory, sorted_events=sorted_events)

    print('Completed data cleaning and generated sorted_events.csv file \n')


    ## split the dataset into train and test
    total_sessions=sorted_events.customer_id.unique()
    np.random.shuffle(total_sessions)

    fractions = np.array([0.8, 0.1, 0.1])
    # split into 3 parts
    train_ids, val_ids, test_ids = np.array_split(
        total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int))

    train_sessions=sorted_events[sorted_events['customer_id'].isin(train_ids)]
    val_sessions=sorted_events[sorted_events['customer_id'].isin(val_ids)]
    test_sessions=sorted_events[sorted_events['customer_id'].isin(test_ids)]

    to_pickled_df(data_directory, sampled_train=train_sessions)
    to_pickled_df(data_directory, sampled_val=val_sessions)
    to_pickled_df(data_directory,sampled_test=test_sessions)

    print('Sampled the dataset for training and test \n')
    print("\nScript completed successfully!")

