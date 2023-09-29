#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH -p longrun
#SBATCH --time 6-24:00:00
#SBATCH --mem 20G
#SBATCH --constraint a6000
#SBATCH --output pretrain_s1s2_.out
 
setcuda 11.0
conda activate myenv
PAIRS=no
AUGS=no
NEGS=yes
MODEL=dual
SCENARIO=s1s2
REGULARIZE=no
DATA_RATIO=1.0
CLIP_SAMPLE_VALUE=yes
BATCH_SIZE=200
TRAIN_EPOCHS=200
# Leave empty to pre-train the corresponding model from scratch
MODEL_PATH=${1-""}
# MODEL_PATH="model_shared_200ep_pairs_5.pckl"
# MODEL_PATH="random"

# Leave empty to re-use the same model
VERSION=""


# Set to 1 to retrain the model anyway
CLEAN=0

if [[ "$MODEL_PATH" == "" ]]
then
  ARGS=""
  DESC="_${MODEL}_${DATA_RATIO}"
  case "$PAIRS" in
    yes)
      ARGS="$ARGS --use_pairs"
      DESC="${DESC}_pairs"
    ;;
    no)
      DESC="${DESC}_nopairs"
    ;;
  esac


  case "$AUGS" in
    no)
      ARGS="$ARGS --no_augs"
      DESC="${DESC}_noaugs"
    ;;
    yes)
      DESC="${DESC}_augs"
    ;;
  esac

  case "$NEGS" in
    yes)
      ARGS="$ARGS --use_negative_labels"
      DESC="${DESC}_negs"
    ;;
    no) ;;
  esac


  case "$REGULARIZE" in
    yes)
      ARGS="$ARGS --regularize_after_block"
      DESC="${DESC}_regul"
    ;;
    no) ;;
  esac

  case "$VERSION" in
    "") ;;
    *) VERSION="_${VERSION}" ;;
  esac

  $var_img = "_"
  if [ "$DATASET" = sen12ms ]; then
    var=$IMAGE_PX_SIZE
  fi


  OUT_FILE="train_encoder${DESC}_${SCENARIO}_${DATASET}_${var_img}_${VERSION}.out"
  echo "bash: OUT_FILE = '$OUT_FILE'"

  if [[ -f "$OUT_FILE" ]]
  then
    LAST_LINE=$(tail --lines=1 $OUT_FILE)

    if grep -q ">> save model at " $OUT_FILE
    then
      echo "bash: Searching in $OUT_FILE"
      MODEL_PATH=$(echo $LAST_LINE | tr " " "\n" | tail --lines=1)
    fi
  fi
else
  OUT_FILE="train_$(basename -s .pckl $MODEL_PATH).out"
fi

if [[ "$MODEL_PATH" == "random" ]]
then
  echo "bash: Evaluating with random weights"
elif [[ ! -f "$MODEL_PATH" || "$CLEAN" == 1 ]]
then
  
  python -u meterml_pretrain.py \
    --n_epochs $TRAIN_EPOCHS \
    --model "$MODEL" \
    --scenario $SCENARIO \
    --data_ratio $DATA_RATIO \
    --batch_size $BATCH_SIZE \
    $ARGS | tee $OUT_FILE

  SAVED_MODEL_PATH=$(tail --lines=1 $OUT_FILE | tr " " "\n" | tail --lines=1)
  if [[ "$SAVED_MODEL_PATH" != "$MODEL_PATH" && "$MODEL_PATH" != "" ]]
  then
    mv $SAVED_MODEL_PATH $MODEL_PATH
  else
    MODEL_PATH="$SAVED_MODEL_PATH"
  fi

  echo "bash: Done pre-training (model saved to $MODEL_PATH)"
else
  echo "bash: Skipping training ($MODEL_PATH exists)"
fi

