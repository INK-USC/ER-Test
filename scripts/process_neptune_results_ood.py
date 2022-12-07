import os, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def main(args):
    if "," in args.exp_ids:
        args.exp_ids = args.exp_ids.split(",")
    else:
        args.exp_ids = args.exp_ids.split(" ")
    args.exp_ids = ["HER-"+x.strip() for x in args.exp_ids]
    data_path = os.path.join(args.data_dir, args.dataset, f'{args.dataset}_{args.filename}.csv')
    df = pd.read_csv(data_path)

    col_list = list(df.columns)
    for key in ['Creation Time', 'Owner']:
        col_list.remove(key)
    df = df.drop_duplicates(subset=col_list)
   
    remove_list = [
        'logs/test_best_perf (last)', 'logs/dev_best_perf (last)',
        'logs/epoch (last)', 'logs/best_epoch (last)', 
        'parameters/seed', 'parameters/data/num_train_seed', 'parameters/data/pct_train_rationales_seed',
    ]
    for key in remove_list:
        if key in col_list:
            col_list.remove(key)

    df = df.fillna(-1)

    group_info = df.loc[df['Id'].isin(args.exp_ids)]
    print(group_info)

    test_perf = group_info['logs/test_acc_metric_epoch (average)']

    vals = list(test_perf.values)

    print(vals)

    mean_test_perf = np.mean(test_perf)
    std_test_perf = np.std(test_perf, ddof=1)

    print(f'test perf: {mean_test_perf:.2f} +/- {std_test_perf:.2f}\n')
    # for x in group_info:
    #     print(x)
    #     print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/neptune/')
    parser.add_argument('--dataset', type=str, choices=['sst', 'cose', 'movies', 'multirc', 'esnli', 'yelp', 'amazon', 'stf', 'olid', 'irony', 'cmo', 'cmc', 'contrast_mnli', 'stf', 'gab', 'tweetevalhate', 'imdb'])
    parser.add_argument('--filename', type=str, choices=['lm', 'er', 'fresh', 'attr'])
    parser.add_argument('--exp_ids',type=str)
    args = parser.parse_args()
    main(args)