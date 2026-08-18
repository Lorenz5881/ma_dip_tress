#!/bin/bash

# Define the directory for the virtual environment and the requirements file
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
start=$(date +%s)
PREPROCESS="${PREPROCESS:-0}"
STRAIN="${STRAIN:-A_PuertoRico_8_1934}"

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
PREPROCESS="${PREPROCESS:-$PREPROCESS}"
STRAIN="${STRAIN:-$STRAIN}"

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
CUTOFF_PAIRS=( # Define cutoffs and respective names
  "15:c15"
  "10:c10"
  "5:c5"
  "0:c0"
)
r_value=$RANDOM
# Run the Python script
cd src/Classifier\ base/

if [ "$PREPROCESS" -eq 1 ]; then  # start with preprocessing if PREPROCESS is set to 1
  echo "Preprocessing data for strain $STRAIN..."
  nohup python ./preprocess_data.py --recluster --strain "$STRAIN" > "preprocess_${STRAIN}.out" 2>&1 &
  wait # Wait for preprocessing to finish before starting model checks
fi

# Intersection prediction for benchmark and relational models
echo "Running intersection predictions for strain $STRAIN..."
for cutoff in "${CUTOFF_PAIRS[@]}"; do
  cutoff_value="${cutoff%:*}"
  cutoff_name="${cutoff#*:}"
  echo "Submitting job for cutoff $cutoff_value ($cutoff_name)..."
  nohup python ./prediction_test.py -g -d ${STRAIN} --package benchmark -i 0 -o ${cutoff_value} -m clf --test_type "intersection" > "intersection_clf_benchmark_${cutoff_value}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i 0 -o ${cutoff_value} -m clf --test_type "intersection" > "intersection_clf_relational_${cutoff_value}.out" 2>&1 &
done

######## Single-dataset prediction testing ########
# looping over publication IDs 1-7 for A/PuertoRico/8/1934
# To compare the difference between keeping and dropping intersecting DelVGs for sum pooled data
echo "Running single-dataset sum pooled benchmarks with and without intersectings..."
for pub_id in {0..6}; do
  echo "Submitting job for publication ID $pub_id..."
  nohup python ./prediction_test.py -gvx -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m reg --test_type "single_dataset" > "single_reg_benchmark_sum_drop_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m reg --test_type "single_dataset" > "single_reg_benchmark_sum_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gvx -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m clf --test_type "single_dataset" > "single_clf_benchmark_sum_drop_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m clf --test_type "single_dataset" > "single_clf_benchmark_sum_keep_${pub_id}.out" 2>&1 &
done

# Using relational intersecting features on sum pooled data
echo "Running single-dataset sum pooled relational models..."
for pub_id in {0..6}; do
  echo "Submitting job for publication ID $pub_id..."
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -p sum -m reg --test_type "single_dataset" > "single_reg_relational_sum_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -p sum -m clf --test_type "single_dataset" > "single_clf_relational_sum_keep_${pub_id}.out" 2>&1 &
done

# Getting results for unpooled data to compare to pooled
echo "Running single-dataset unpooled relational and benchmark models..."
for pub_id in {0..6}; do
  echo "Submitting job for publication ID $pub_id..."
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -m reg --test_type "single_dataset" > "single_reg_benchmark_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -m reg --test_type "single_dataset" > "single_reg_relational_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -m clf --test_type "single_dataset" > "single_clf_benchmark_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -m clf --test_type "single_dataset" > "single_clf_relational_unpooled_keep_${pub_id}.out" 2>&1 &
done
wait # Wait for all single-dataset model checks to finish before starting leave-one-out

######## Leave-one-out prediction testing ########
# leave-one-out benchmark on pooled data
echo "Running leave-one-out predictions on sum pooled data..."
for pub_id in {0..6}; do
  echo "Submitting job for publication ID $pub_id..."
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m reg --test_type "leave_one_out" > "loo_reg_benchmark_sum_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -p sum -m reg --test_type "leave_one_out" > "loo_reg_relational_sum_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m clf --test_type "leave_one_out" > "loo_clf_benchmark_sum_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -p sum -m clf --test_type "leave_one_out" > "loo_clf_relational_sum_keep_${pub_id}.out" 2>&1 &
done

# leave-one-out benchmark on unpooled data
echo "Running leave-one-out predictions on unpooled data..."
for pub_id in {0..6}; do
  echo "Submitting job for publication ID $pub_id..."
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -m reg --test_type "leave_one_out" > "loo_reg_benchmark_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -m reg --test_type "leave_one_out" > "loo_reg_relational_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -m reg --test_type "leave_one_out" > "loo_reg_context_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package benchmark -i "$pub_id" -m clf --test_type "leave_one_out" > "loo_clf_benchmark_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package relational -i "$pub_id" -m clf --test_type "leave_one_out" > "loo_clf_relational_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -m clf --test_type "leave_one_out" > "loo_clf_context_unpooled_keep_${pub_id}.out" 2>&1 &
done

wait # Wait for all leave-one-out model checks to finish before starting single-dataset checks (42 jobs submitted)

######## Miscellaneous NGS prediction testing with context features########
# looping over publication IDs 1-7 for A/PuertoRico/8/1934
# leave-one-out benchmark on pooled data
#echo "Running model check benchmark dropped inter leave-one-out on pooled..."
echo "Running model check benchmark dropped inter leave-one-out on pooled with stratified undersampling for regression..."
for pub_id in {0..6}; do
  echo "Submitting job for publication ID $pub_id..."
  nohup python ./prediction_test.py -gvx -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m reg --test_type "leave_one_out" > "loo_reg_benchmark_sum_drop_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -p sum -m reg --test_type "leave_one_out" > "loo_reg_context_sum_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gvx -d ${STRAIN} --package benchmark -i "$pub_id" -p sum -m clf --test_type "leave_one_out" > "loo_clf_benchmark_sum_drop_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -p sum -m clf --test_type "leave_one_out" > "loo_clf_context_sum_keep_${pub_id}.out" 2>&1 &
done
wait # Wait for all leave-one-out model checks to finish before starting single-dataset checks (42 jobs submitted)

# Getting benchmark results for unpooled data to compare to pooled
echo "Running miscellaneous single-datasets prediction tests with context features..."
for pub_id in {0..6}; do
  echo "Submitting job for publication ID $pub_id..."
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -p sum -m reg --test_type "single_dataset" > "single_reg_context_pooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -p mean -m reg --test_type "single_dataset" > "single_reg_context_pooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -m reg --test_type "single_dataset" > "single_reg_context_unpooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -p sum -m clf --test_type "single_dataset" > "single_clf_context_pooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -p mean -m clf --test_type "single_dataset" > "single_clf_context_pooled_keep_${pub_id}.out" 2>&1 &
  nohup python ./prediction_test.py -gv -d ${STRAIN} --package context -i "$pub_id" -m clf --test_type "single_dataset" > "single_clf_context_unpooled_keep_${pub_id}.out" 2>&1 &
done
echo "All jobs submitted."
wait

end=$(date +%s)
echo "Done. Time elapsed: $((end - start)) seconds."

# Deactivate the virtual environment
deactivate
