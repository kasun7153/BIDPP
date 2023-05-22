from propy import PyPro
import pandas as pd
import numpy as np

# Define directories for data
dir = 'Data/'
raw_data_dir = 'raw_data'
processed_data_dir = 'processed_data'

# Function to extract protein sequences from a FASTA file
def extractProteinSequenceFromFasta(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    protein_sequences = lines[1::2]
    protein_ids = lines[::2]
    return protein_ids, protein_sequences

# Function to extract features and create a DataFrame
def extractFeatureDF(protein_ids, protein_sequences, feature_type, negative):
    df = pd.DataFrame()
 
    for i in range(len(protein_sequences)):
        try:
            protein = PyPro.GetProDes(protein_sequences[i])
            if feature_type == 'AAC':
                extractedFeatures = protein.GetAAComp()

            df1 = pd.DataFrame.from_dict(extractedFeatures, orient='index').transpose()
            df1['id'] = protein_ids[i][1:]
            df = pd.concat([df, df1], ignore_index=True)
        except ZeroDivisionError:
            print(f"Skipping sequence {i} due to ZeroDivisionError")
            continue
    if negative:
        df['label'] = 0
    else:
        df['label'] = 1
    return df

# Function to combine negative and positive datasets into a single DataFrame
def combineNegativeAndPositiveDFs(negativeFile, positiveFile, feature_type):
    # Extract protein sequences and IDs from FASTA files
    negative_ids, negative_sequences = extractProteinSequenceFromFasta(negativeFile)
    positive_ids, positive_sequences = extractProteinSequenceFromFasta(positiveFile)

    # Extract features and create DataFrames for negative and positive datasets
    negativeDF = extractFeatureDF(negative_ids, negative_sequences, feature_type, negative=True)
    positiveDF = extractFeatureDF(positive_ids, positive_sequences, feature_type, negative=False)

    # Combine negative and positive DataFrames
    combinedDF = pd.concat([negativeDF, positiveDF], ignore_index=True)

    # Shuffle the rows of the combined DataFrame
    combinedDF = combinedDF.sample(frac=1).reset_index(drop=True)

    return combinedDF

# Main function to run the code
def run(TR_pos, TR_neg, TS_pos, TS_neg):
    # Combine negative and positive datasets for training data and save as CSV
    combineNegativeAndPositiveDFs(f'{dir + raw_data_dir}/{TR_neg}', f'{dir + raw_data_dir}/{TR_pos}', 'AAC').to_csv(f'{dir + processed_data_dir}/TR_AAC.csv', index=False)
    # Combine negative and positive datasets for testing data and save as CSV
    combineNegativeAndPositiveDFs(f'{dir + raw_data_dir}/{TS_neg}', f'{dir + raw_data_dir}/{TS_pos}', 'AAC').to_csv(f'{dir + processed_data_dir}/TS_AAC.csv', index=False)
