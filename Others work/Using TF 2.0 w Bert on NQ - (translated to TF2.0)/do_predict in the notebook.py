if FLAGS.do_predict:
  if not FLAGS.output_prediction_file:
    raise ValueError(
        "--output_prediction_file must be defined in predict mode.")
    
  eval_examples = tf2baseline.read_nq_examples(
      input_file=FLAGS.predict_file, is_training=False)

  print("FLAGS.predict_file", FLAGS.predict_file)

  eval_writer = tf2baseline.FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
      is_training=False)
  eval_features = []

  def append_feature(feature):
    eval_features.append(feature)
    eval_writer.process_feature(feature)

  num_spans_to_ids = tf2baseline.convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      is_training=False,
      output_fn=append_feature)
  eval_writer.close()
  eval_filename = eval_writer.filename

  print("***** Running predictions *****")
  print(f"  Num orig examples = %d" % len(eval_examples))
  print(f"  Num split examples = %d" % len(eval_features))
  print(f"  Batch size = %d" % FLAGS.predict_batch_size)
  for spans, ids in num_spans_to_ids.items():
    print(f"  Num split into %d = %d" % (spans, len(ids)))

  predict_input_fn = tf2baseline.input_fn_builder(
      input_file=eval_filename,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

  all_results = []

  for result in estimator.predict(
      predict_input_fn, yield_single_examples=True):
    if len(all_results) % 1000 == 0:
      print("Processing example: %d" % (len(all_results)))

    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]

    all_results.append(
        tf2baseline.RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits,
            answer_type_logits=answer_type_logits))

  print ("Going to candidates file")

  candidates_dict = tf2baseline.read_candidates(FLAGS.predict_file)

  print ("setting up eval features")

  raw_dataset = tf.data.TFRecordDataset(eval_filename)
  eval_features = []
  for raw_record in raw_dataset:
    eval_features.append(tf.train.Example.FromString(raw_record.numpy()))
    
  print ("compute_pred_dict")

  nq_pred_dict = tf2baseline.compute_pred_dict(candidates_dict, eval_features,
                                   [r._asdict() for r in all_results])
  predictions_json = {"predictions": list(nq_pred_dict.values())}

  print ("writing json")

  with tf.io.gfile.GFile(FLAGS.output_prediction_file, "w") as f:
    json.dump(predictions_json, f, indent=4)