#!/bin/bash

# Change to the root directory of the repository to ensure the script runs correctly
cd ../..

# Define the directory for the virtual environment and the requirements file
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
start=$(date +%s)

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found. Creating one..."
  python3 -m venv $VENV_DIR
fi
# Activate the virtual environment
source $VENV_DIR/bin/activate

# Check if the requirements file exists and install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing dependencies..."
  pip install -r $REQUIREMENTS_FILE -q
fi


# Getting back to the parsing directory
cd src/srr_to_unpooled/

# Run the parsing script to create the parsed data files
echo "Parsing raw data..."
python ./parse_to_csv.py > parsing.log 2>&1
echo "Parsing complete."

# Run the testing script to compare the parsed data to given sets
echo "Testing the parsed data..."
python ./test_parsed.py > testing.log 2>&1
echo "Testing complete."

end=$(date +%s)
echo "Finished parsing and testing the result."
echo "Time elapsed: $((end - start)) seconds."

# Deactivate the virtual environment
deactivate
