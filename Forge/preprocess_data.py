import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import re

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def create_chat_format(user_query, bot_response):
    return f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n{bot_response}<|im_end|>"

def main():
    data_path = "D:\\Programming\\Hackathons\\2026\\Goolge Deepmind X Devpost Hackathon\\Forge\\english_support_dataset (1).csv"
    output_dir = "./data"

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("Trying 'latin1' encoding...")
        df = pd.read_csv(data_path, encoding='latin1')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Data loaded successfully.")
    print(f"Original shape: {df.shape}")

    print("Handling missing values...")
    df = df.dropna(subset=['user_query_en', 'bot_response_en'])

    # Impute missing categories with "unknown"
    df['category'] = df['category'].fillna('unknown')
    print(f"Shape after dropping NA: {df.shape}")

    print("Removing duplicates...")
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")

    print("Preprocessing text...")
    df['user_query_en'] = df['user_query_en'].apply(preprocess_text)
    df['bot_response_en'] = df['bot_response_en'].apply(preprocess_text)

    print("Creating 'text' column...")
    df['text'] = df.apply(lambda row: create_chat_format(row['user_query_en'], row['bot_response_en']), axis=1)

    print("Splitting into train/validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    print(f"Train set size: {train_df.shape}")
    print(f"Validation set size: {val_df.shape}")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "processed_train.jsonl")
    val_file = os.path.join(output_dir, "processed_val.jsonl")

    print(f"Saving train data to {train_file}...")
    try:
        with open(train_file, 'w', encoding='utf-8') as f:
            for _, row in train_df.iterrows():
                json.dump({"text": row['text']}, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        print(f"Error saving train data: {e}")
        return
    print("Train data saved successfully.")

    print(f"Saving validation data to {val_file}...")
    try:
        with open(val_file, 'w', encoding='utf-8') as f:
            for _, row in val_df.iterrows():
                json.dump({"text": row['text']}, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        print(f"Error saving validation data: {e}")
        return
    print("Validation data saved successfully.")

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()