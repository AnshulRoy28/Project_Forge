import pandas as pd
import json
import os
import random
from pathlib import Path

def log_progress(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

# Define the configuration
CONFIG = {
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "architecture": "llama",
    "chat_template": "llama",
    "max_length": 2048,
    "dataset_path": "test_data.csv",
    "train_split": 0.9,
    "validation_split": 0.1,
    "chunk_size": 1000,
    "min_text_length": 10,
    "max_text_length": None,
    "remove_duplicates": True,
    "random_seed": 42,
    "output_dir": "/output"
}

def load_and_prepare_data(dataset_path, remove_duplicates=True):
    """Loads the dataset, handles missing values, and removes duplicates."""
    try:
        log_progress(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path, encoding='utf-8')

        # Handle missing values (fill with empty string)
        df = df.fillna('')

        # Remove duplicates
        if remove_duplicates:
            log_progress("Removing duplicate rows")
            df = df.drop_duplicates()

        log_progress(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except FileNotFoundError:
        log_error(f"File not found: {dataset_path}")
        return None
    except Exception as e:
        log_error(f"Error loading dataset: {e}")
        return None

def apply_chat_template(df):
    """Applies the chat template to the query and response columns."""
    try:
        log_progress("Applying chat template")
        df["text"] = df.apply(
            lambda row: "<s>[INST] {query} [/INST] {response} </s>".format(
                query=row["query"], response=row["response"]
            ),
            axis=1,
        )
        return df
    except Exception as e:
        log_error(f"Error applying chat template: {e}")
        return None

def filter_by_text_length(df, min_length=10, max_length=None):
    """Filters the DataFrame based on the length of the 'text' column."""
    try:
        log_progress(f"Filtering by text length (min={min_length}, max={max_length})")
        df['text_length'] = df['text'].apply(len)
        df = df[df['text_length'] >= min_length]
        if max_length:
            df = df[df['text_length'] <= max_length]
        df = df.drop('text_length', axis=1)
        return df
    except Exception as e:
        log_error(f"Error filtering by text length: {e}")
        return None

def split_data(df, train_split=0.9, val_split=0.1, random_seed=42):
    """Splits the DataFrame into training and validation sets."""
    try:
        log_progress(f"Splitting data into train ({train_split}) and validation ({val_split}) sets")
        random.seed(random_seed)
        train_size = int(train_split * len(df))
        train_indices = random.sample(df.index.tolist(), train_size)
        train_df = df.loc[train_indices]
        val_df = df.drop(train_indices)
        return train_df, val_df
    except Exception as e:
        log_error(f"Error splitting data: {e}")
        return None, None

def save_data(df, output_path):
    """Saves the DataFrame to a JSONL file."""
    try:
        log_progress(f"Saving data to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                json.dump({"text": row["text"]}, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        log_error(f"Error saving data to {output_path}: {e}")

def main():
    """Main function to orchestrate the data preprocessing."""
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(CONFIG["dataset_path"], CONFIG["remove_duplicates"])
    if df is None:
        return

    df = apply_chat_template(df)
    if df is None:
        return

    df = filter_by_text_length(df, CONFIG["min_text_length"], CONFIG["max_text_length"])
    if df is None:
        return

    train_df, val_df = split_data(df, CONFIG["train_split"], CONFIG["validation_split"], CONFIG["random_seed"])
    if train_df is None or val_df is None:
        return

    save_data(train_df, os.path.join(CONFIG["output_dir"], "processed_train.jsonl"))
    save_data(val_df, os.path.join(CONFIG["output_dir"], "processed_val.jsonl"))

    log_progress("Data preprocessing complete.")

if __name__ == "__main__":
    main()