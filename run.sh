TASK=music
OUTPUT=experiment/bert/music
DATA=data/${TASK}

# Train and eval
CUDA_VISIBLE_DEVICES=2 python model.py --task_name=${TASK} --data_dir=${DATA} --output_dir=${OUTPUT} \
    --do_train --do_eval --num_train_epochs=18 --lr=3e-5

# Run Score and post-processing with MED
echo "Without MED"
python scripts/score.py --annotation ${DATA}/development.json --prediction ${OUTPUT}/write_results.json

echo "With MED"
python med.py --output_dir=${OUTPUT} --ontology_path=${DATA}/ontology.json
python scripts/score.py --annotation ${DATA}/development.json --prediction ${OUTPUT}/write_results_med.json
