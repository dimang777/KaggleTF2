

# evaluate
python -m language.question_answering.bert_joint.run_nq \
  --logtostderr \
  --bert_config_file=bert-joint-baseline/bert_config.json \
  --vocab_file=bert-joint-baseline/vocab-nq.txt \
  --predict_file=tiny-dev/nq-dev-sample.no-annot.jsonl.gz \
  --init_checkpoint=bert-joint-baseline/bert_joint.ckpt \
  --do_predict \
  --output_dir=bert_model_output \
  --output_prediction_file=bert_model_output/predictions.json





# train
python -m language.question_answering.bert_joint.run_nq \
  --logtostderr \
  --bert_config_file=bert-joint-baseline/bert_config.json \
  --vocab_file=bert-joint-baseline/vocab-nq.txt \
  --train_precomputed=nq-train.tfrecords-00000-of-00001 \
  --train_num_precomputed=494670 \
  --learning_rate=3e-5 \
  --num_train_epochs=1 \
  --max_seq_length=512 \
  --save_checkpoints_steps=5000 \
  --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --do_train \
  --output_dir=bert_model_output