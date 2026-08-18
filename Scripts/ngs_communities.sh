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
CUTOFF_PAIRS=( # Define cutoffs and respective names
  "15:c15"
  "10:c10"
  "5:c5"
  "0:c0"
)
r_value=$RANDOM
# Run the Python script
cd "$PROJECT_DIR/src/clustering/"

nohup python ngs_clustering.py --strain A_WSN_33 --unpooled --cutoff 0 --community-methods leiden louvain --leiden-partition modularity cpm rbconfiguration rber surprise significance --leiden-signed-extra-partitions cpm --run-name single_call_all_partitions > "A_WSN_33_ngs_clustering.out" 2>&1 &

nohup python ngs_clustering.py --strain B_Victoria_504_2000 --unpooled --cutoff 0 --community-methods leiden louvain --leiden-partition modularity cpm rbconfiguration rber surprise significance --leiden-signed-extra-partitions cpm --run-name single_call_all_partitions > "B_Victoria_504_2000_ngs_clustering.out" 2>&1 &

nohup python ngs_clustering.py --strain B_Yamagata_16_1988 --unpooled --cutoff 0 --community-methods leiden louvain --leiden-partition modularity cpm rbconfiguration --leiden-signed-extra-partitions cpm --run-name single_call_all_partitions > "B_Yamagata_16_1988_ngs_clustering.out" 2>&1 &

nohup python ngs_clustering.py --strain A_PuertoRico_8_1934 --unpooled --cutoff 0 --community-methods leiden louvain --leiden-partition modularity cpm rbconfiguration --leiden-signed-extra-partitions cpm --run-name single_call_all_partitions > "A_PuertoRico_8_1934_ngs_clustering.out" 2>&1 &

echo "All jobs submitted."
wait
end=$(date +%s)
echo "Time elapsed: $((end - start)) seconds."

# Deactivate the virtual environment
deactivate
