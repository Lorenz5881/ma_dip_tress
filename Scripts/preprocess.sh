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
STRAIN="${STRAIN:-A_PuertoRico_8_1934}" # default strain to preprocess if not provided
# B_Yamagata_16_1988}"

for arg in "$@"; do
  case "$arg" in
    *=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      export "$key=$value"
      ;;
    *)
      echo "Ignoring unsupported argument '$arg'. Use KEY=VALUE overrides."
      ;;
  esac
done
STRAIN="${STRAIN:-$STRAIN}"

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

# do actual preprocessing for the specified strain
cd "$PROJECT_DIR/src/prediction/"
echo "Preprocessing data for strain $STRAIN..."
nohup python ./preprocess_data.py --strain "$STRAIN" > "preprocess_${STRAIN}.out" 2>&1 &
wait

end=$(date +%s)
echo "All jobs submitted."
echo "Time elapsed: $((end - start)) seconds."

# Deactivate the virtual environment
deactivate
