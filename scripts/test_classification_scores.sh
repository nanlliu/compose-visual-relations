output_folder='./'
data_folder='./dataset'

for dataset in "igibson" "clevr"
  do
    for num_rels in 1 2 3
      do
        FILEPATH="${output_folder}/${dataset}"
        command="python classification_scores.py --dataset ${dataset} --checkpoint_dir ./binary_classifier/ \
        --data_folder ${data_folder} --generated_img_folder ${FILEPATH}/num_rel_${num_rels} \
        --mode generation --num_rels ${num_rels}"
        $command
      done
  done