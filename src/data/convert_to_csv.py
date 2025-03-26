import pandas as pd
import os

RAW_DATA_PATH = "../../data/raw/monolingual-n/" 
PROCESSED_DATA_PATH = "../../src/data/preocessed/processed_IITB.csv"


def load_raw_dakshina_data(path):
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
    return data


raw_data = load_raw_dakshina_data(RAW_DATA_PATH)
print(raw_data[:50])


def save_processed_data(data, path):
    df = pd.DataFrame({"text": data}) #converting the list to df
    os.makedirs(os.path.dirname(path), exist_ok=True) # - if file already exist (to deal with error)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Processed data saved! @ {path}")


#to save the dataset into csv file.
save_processed_data(raw_data, PROCESSED_DATA_PATH)