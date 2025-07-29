import os
import numpy as np
import utils.tokenization as tokenization


def build_dataset(data_dir, output_file):
    """
    Builds a dataset from the files in the specified directory and saves it to an output file.

    Parameters:
    - data_dir (str): The directory containing the data files.
    - output_file (str): The file where the dataset will be saved.
    """
    dataset = []
    # sperator token ID to separate different text entries
    sep_id = tokenization.text_to_ids('[SEP]')

    # Iterate through all files in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):  # Assuming text files for simplicity
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Convert tokens to IDs
                token_ids = tokenization.text_to_ids(content)
                token_ids.append(sep_id[0])
                # extend the dataset with tokenized content and IDs
                dataset.extend(token_ids)
                
    # Convert dataset to a numpy array and save it
    print(dataset)
    np.save(output_file, np.array(dataset))
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    data_directory = "scraped_articles"  # Directory containing text files
    output_file_path = "dataset/dataset.npy"  # Output file for the dataset
    build_dataset(data_directory, output_file_path)
    print("Dataset building completed.")