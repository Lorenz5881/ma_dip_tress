#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$SCRIPT_DIR/.venv" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../.venv" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "Could not find .venv"
    exit 1
fi

VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
start=$(date +%s)
THRESHOLD=3

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found. Creating one..."
  python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    echo "Could not find virtual environment activation script."
    exit 1
fi

# Check if the requirements file exists and install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing dependencies..."
  pip install -r $REQUIREMENTS_FILE -q
fi

STRAINS=(
  "A_PuertoRico_8_1934"
  "B_Yamagata_16_1988"
  "B_Victoria_504_2000"
  "A_WSN_33"
)

#start_time=$(date +"%Y-%m-%d_%H-%M-%S")
start_time="arts"
echo "starting jobs at $start_time"
cd "$PROJECT_DIR/src/clustering/"
nohup python ./feature_scaffolds.py -e -t 1 > na_pr8.out 2>&1 
#for o in "${STRAINS[@]}"; do
  #nohup python ./feature_scaffolds.py -e -s "$o" > loaded_feature_scaffolds.out 2>&1 
  #nohup python ./feature_scaffolds.py -s "$o" > standard_feature_scaffolds.out 2>&1 
  #nohup python ./feature_scaffolds.py -x -n "noSeg_${o}" -s "$o" > noSeg_feature_scaffolds.out 2>&1 
#done
deactivate
wait
# Calculate the time elapsed
end=$(date +%s)
echo "All jobs completed."
echo "Time elapsed: $((end - start)) seconds."
