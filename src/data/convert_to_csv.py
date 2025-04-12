import pandas as pd
import os

def load_raw_data(input_folder_path, output_csv_path="./hindi_corpus.csv"):
    """
    Loads .hi files from a folder and saves the combined lines into a CSV.

    Parameters:
    - input_folder_path: str → path to folder containing .hi files
    - output_csv_path: str → output path for the generated CSV
    """

    data = []
    for file in os.listdir(input_folder_path):
        if file.endswith(".hi"):
            file_path = os.path.join(input_folder_path, file)
            print(f"Reading: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data.extend(f.readlines())
        else:
            print(f"Skipping: {file}")
    
    print(f"Loaded {len(data)} lines.")

    df = pd.DataFrame({"text": data})
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Processed data saved at: {output_csv_path}")

# Optional: run directly
if __name__ == "__main__":
    load_raw_data(
        input_folder_path="../../data/raw/monolingual-n/",
        output_csv_path="../../data/raw/monolingual-n/raw_IITB.csv"
    )