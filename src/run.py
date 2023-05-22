import sys
import os
import data_extraction as de
import train as model

# Checking if the 'out' directory exists or not. If not, create it.
if not os.path.exists("out"):
    os.makedirs("out")

# Checking if the 'Data/processed_data' directory exists or not. If not, create it.
if not os.path.exists("Data/processed_data"):
    os.makedirs("Data/processed_data")

# Check if the command line arguments are provided
if len(sys.argv) < 4:
    print("Usage: python3 run.py <positive training data> <negative training data> <positive testing data> <negative testing data>")
    exit(1)

# Retrieve the command line arguments
TR_pos = sys.argv[1]
TR_neg = sys.argv[2]
TS_pos = sys.argv[3]
TS_neg = sys.argv[4]

# Run data extraction
de.run(TR_pos, TR_neg, TS_pos, TS_neg)

# Run the training model
model.run()
