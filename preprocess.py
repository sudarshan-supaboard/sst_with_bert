import kagglehub
import pandas as pd

from datasets import Dataset, DatasetDict
from pathlib import Path
from pprint import pprint

from config import Config
# Download latest version

path = kagglehub.dataset_download(Config.DATASET_PATH)

print("Path to dataset files:", path)

ds_path = Path(path)
emotions = ds_path / 'emotions.txt'
train_path = ds_path / 'train.csv'
test_path = ds_path / 'test.csv'
valid_path = ds_path / 'val.csv'
    

def get_label_dict():
    with open(emotions) as f:
        idx_to_labels = f.readlines()

    idx_to_labels = list(map(lambda x: x.strip(), idx_to_labels))
    
    if len(idx_to_labels) != Config.NUM_LABELS:
        raise Exception("number of labels in config is not consitent")
    
    # create labels_to_idx
    labels_to_idx = {}
    for i, label in enumerate(idx_to_labels):
        labels_to_idx[label] = i
    
    return idx_to_labels , labels_to_idx


idx_to_labels, labels_to_idx = get_label_dict()

def make_datasets():
    
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    valid_df = pd.read_csv(valid_path)
    
    
    
    # shuffle the train_df
    train_df = train_df.sample(frac=1, ignore_index=True, random_state=Config.RANDOM_STATE)
    test_df = test_df.sample(frac = 1, ignore_index=True, random_state=Config.RANDOM_STATE)
    valid_df = valid_df.sample(frac = 1, ignore_index=True, random_state=Config.RANDOM_STATE)

    train_df['label'] = train_df['label'].apply(lambda x: labels_to_idx[x])
    test_df['label'] = test_df['label'].apply(lambda x: labels_to_idx[x])
    valid_df['label'] = valid_df['label'].apply(lambda x: labels_to_idx[x])

    pprint({
        'train_df.shape': train_df.shape,
        'test_df.shape': test_df.shape,
        'valid_df.shape': valid_df.shape,
    })

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset, "test": test_dataset})
    
    return dataset


if __name__ == '__main__':
    dataset = make_datasets()
    pprint(dataset['train'][:4])
    pprint(labels_to_idx)