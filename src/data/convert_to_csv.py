import pandas as pd
import os

def load_raw_data(path):
    data = []
    for file in os.listdir(path):
        if file.endswith(".hi"): #open txt file
            file_path = os.path.join(path, file)
            print(f"Reading: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data.extend(f.readlines())  #read all the lines of file
        else:
            print(f"Skipping: {file}")
    print(f"Loaded {len(data)} lines")

    PROCESSED_DATA_PATH = "./hindi_corpus.csv"

    df = pd.DataFrame({"text": data}) #converting the list to df
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True) # - if file already exist (to deal with error)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8")
    print(f"Processed data saved! @ {PROCESSED_DATA_PATH}")