# NUM_RELS range from 1 to 3
# data folder is the directory in which you place the numpy data file
python inference.py --checkpoint_folder ./checkpoints --model_name clevr_test \
--output_folder $YOUR_IMG_FOLDER --dataset clevr --resume_iter $YOUR_RESUME_ITERATION \
--batch_size 32 --num_steps 80 --num_rels $NUM_RELS(1-3) --data_folder ./dataset --mode editing