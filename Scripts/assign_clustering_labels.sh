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

STRAIN="B_Yamagata_16_1988"
cd "$PROJECT_DIR/src/clustering/"
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -a hdbscan > "scaffold pure hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -a kmeans > "scaffold pure kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 0 -a hdbscan > "scaffold 0 hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 0 -a kmeans > "scaffold 0 kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 5 -a hdbscan > "scaffold 5 hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 5 -a kmeans > "scaffold 5 kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 10 -a hdbscan > "scaffold 10 hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 10 -a kmeans > "scaffold 10 kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 15 -a hdbscan > "scaffold 15 hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t scaffold -o 15 -a kmeans > "scaffold 15 kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -a hdbscan > "comb hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -a kmeans > "comb kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 0 -a hdbscan > "comb 0 hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 0 -a kmeans > "comb 0 kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 5 -a hdbscan > "comb 5 hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 5 -a kmeans > "comb 5 kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 10 -a hdbscan > "comb 10 hdbscan ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 10 -a kmeans > "comb 10 kmeans ${STRAIN}.out" 2>&1 &
#nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 15 -a hdbscan > "comb 15 hdbscan ${STRAIN}.out" 2>&1 &
nohup python ./assign_clustering_labels.py -g -s "$STRAIN" -t comb -o 15 -a kmeans > "comb 15 kmeans ${STRAIN}.out" 2>&1 &

wait
end=$(date +%s)
echo "All jobs completed."
echo "Time elapsed: $((end - start)) seconds."
# Deactivate the virtual environment
deactivate
