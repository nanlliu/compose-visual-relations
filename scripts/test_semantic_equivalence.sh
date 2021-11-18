DATA_FOLDER='./dataset'
for num_rels in 1 2 3
  do
    python semantic_equivalence.py --data_folder ${DATA_FOLDER} --dataset clevr --num_rels ${num_rels} --model classifier --checkpoint_folder ./binary_classifier/
    python semantic_equivalence.py --data_folder ${DATA_FOLDER} --num_rels ${num_rels} --checkpoint_folder ../checkpoints/ --model_name clevr_ebm --resume_iter $RESUME_ITER --model ebm-emb --dataset clevr
    python semantic_equivalence.py --data_folder ${DATA_FOLDER} --num_rels ${num_rels} --checkpoint_folder ../checkpoints/ --model_name clevr_ebm_clip --resume_iter $RESUME_ITER --model ebm-clip --dataset clevr
  done