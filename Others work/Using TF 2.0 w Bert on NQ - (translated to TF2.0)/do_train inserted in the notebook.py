bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

validate_flags_or_throw(bert_config)
tf.io.gfile.makedirs(FLAGS.output_dir)

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

run_config = tf.estimator.RunConfig(
  model_dir=FLAGS.output_dir,
   save_checkpoints_steps=FLAGS.save_checkpoints_steps)

num_train_steps = None
num_warmup_steps = None
if FLAGS.do_train:
  num_train_features = FLAGS.train_num_precomputed
  num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                        FLAGS.num_train_epochs)

num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

model_fn = tf2baseline.model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={'batch_size':FLAGS.train_batch_size})


if FLAGS.do_train:
  print("***** Running training on precomputed features *****")
  print("  Num split examples = %d", num_train_features)
  print("  Batch size = %d", FLAGS.train_batch_size)
  print("  Num steps = %d", num_train_steps)
  train_filenames = tf.io.gfile.glob(FLAGS.train_precomputed_file)
  train_input_fn = input_fn_builder(
      input_file=train_filenames,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)