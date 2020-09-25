export CWD=/home/TILDE.LV/rinalds.viksna/BERT/SQuAD_BERT
export BERT_DIR=$CWD/bert
#export MODEL_DIR=/home/TILDE.LV/rinalds.viksna/BERT/models/lv_v2_cased_L-12_H-768_A-12
export MODEL_DIR=$CWD/bert_model

export SQUAD_DIR=$CWD/SQuAD-LV
export OUTPUT_DIR=$CWD/squad_model_lv

export TRAIN_FILE=$SQUAD_DIR/train-v2.0.LV.json
export PREDICT_FILE=$SQUAD_DIR/dev-v2.0.LV.json


CUDA_VISIBLE_DEVICES=0 python $BERT_DIR/run_squad.py \
	--vocab_file=$MODEL_DIR/vocab.txt \
	--bert_config_file=$MODEL_DIR/bert_config.json \
	--init_checkpoint=$MODEL_DIR/bert_model.ckpt \
	--do_lower_case=False \
	--do_train=False \
	--train_file=$TRAIN_FILE \
	--do_predict=True \
	--predict_file=$PREDICT_FILE \
	--train_batch_size=8 \
	--learning_rate=3e-5 \
	--num_train_epochs=2.0 \
	--max_seq_length=384 \
	--doc_stride=128 \
	--output_dir=$OUTPUT_DIR \
	--use_tpu=False \
	--version_2_with_negative=True


# Run above, then run script 
python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.LV.json $OUTPUT_DIR/predictions.json --na-prob-file $OUTPUT_DIR/null_odds.json
