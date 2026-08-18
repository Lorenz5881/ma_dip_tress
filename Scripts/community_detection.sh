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
)
#"B_Yamagata_16_1988"
#"B_Victoria_504_2000"
#"A_WSN_33"

#start_time=$(date +"%Y-%m-%d_%H-%M-%S")
test_name="community_detection"
cd "$PROJECT_DIR/src/clustering/"
nohup python ./community_detection.py -am -d 3 -p "red" -l "test" -n "louvain_vs_leiden" > "louvain_vs_leiden".out 2>&1 &
#nohup python ./community_detection.py -am -d 4 -p "red" -l "test" -n "simple_leiden" > "simple_leiden".out 2>&1 &
#nohup python ./community_detection.py -am -c 15 -d 3 -p "red" -l "test" -n "louvain_vs_leiden_meta" -f "Alnaji2021" "Pelz2021" "Wang2020" "Wang2023" > "louvain_vs_leiden_meta".out 2>&1 &
#nohup python ./community_detection.py -am -c 15 -d 4 -p "red" -l "test" -n "simple_leiden_meta" -f "Alnaji2021" "Pelz2021" "Wang2020" "Wang2023" > "simple_leiden_meta".out 2>&1 &
#nohup python ./community_detection.py -am -d 1 -p "red" -l "test" -n "${test_name}_nan" > "${test_name}_nan".out 2>&1 &
#nohup python ./community_detection.py -am -d 1 -p "red" -z=0 -l "test" -n "${test_name}_0" > "${test_name}_0".out 2>&1 &
#nohup python ./community_detection.py -am -d 1 -p "red" -z=-1 -l "test" -n "${test_name}_-1" > "${test_name}_-1".out 2>&1 &
#nohup python ./community_detection.py -am -d 1 -p "red" -z=-99999 -l "test" -n "${test_name}_max" > "${test_name}_max".out 2>&1 &
#nohup python ./community_detection.py -am -d 1 -p "red" -t "NGS_read_count" -z=0 -l "test" -n "${test_name}_ngs0" > "${test_name}_ngs0".out 2>&1 &

deactivate
wait
# Calculate the time elapsed
end=$(date +%s)
echo "All jobs completed."
echo "Time elapsed: $((end - start)) seconds."
