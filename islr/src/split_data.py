import pandas as pd
import os

def split_data(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    # Shuffle the dataframe to ensure random sampling for test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Group by 'id' and take the first one for test
    test_indices = df.groupby('id').head(1).index
    
    df_test = df.loc[test_indices]
    df_train = df.drop(test_indices)
    
    # Sanity check
    train_ids = set(df_train['id'].unique())
    test_ids = set(df_test['id'].unique())
    
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print(f"Unique words in train: {len(train_ids)}")
    print(f"Unique words in test: {len(test_ids)}")
    
    if not test_ids.issubset(train_ids):
        print("WARNING: Some words in test set are NOT in the training set!")
    else:
        print("Success: All words in test set are present in training set.")
    
    df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    csv_path = "/home/slime-base/projects/jian/islr/data/preprocessed_data_full.csv"
    output_dir = "/home/slime-base/projects/jian/islr/data"
    split_data(csv_path, output_dir)
