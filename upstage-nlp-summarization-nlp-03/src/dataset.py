import pandas as pd
from datasets import Dataset, DatasetDict

def load_data(data_config):
    train_df = pd.read_csv(data_config.train_path)
    valid_df = pd.read_csv(data_config.valid_path)
    test_df = pd.read_csv(data_config.test_path)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    })

    return dataset