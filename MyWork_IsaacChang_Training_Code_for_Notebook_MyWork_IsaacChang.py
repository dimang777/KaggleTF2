# Entire cell added by Isaac
# Code required for training - taken from tf2_0_baseline_w_bert_translated_to_tf2_0

# Must enable flags to enable training including do_train
# Here the flags are set differently from above and indicated as "CHANGED" if it is different than original

# The flags are based on the example train command given in 
# https://github.com/google-research/language/tree/master/language/question_answering/bert_joint
# Copy of the example command taken from the Github repo 
# init_checkpoint should be changed to bert-joint-baseline/bert_joint.ckpt

# python -m language.question_answering.bert_joint.run_nq \
#   --logtostderr \
#   --bert_config_file=bert-joint-baseline/bert_config.json \
#   --vocab_file=bert-joint-baseline/vocab-nq.txt \
#   --train_precomputed=nq-train.tfrecords-00000-of-00001 \
#   --train_num_precomputed=494670 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=1 \
#   --max_seq_length=512 \
#   --save_checkpoints_steps=5000 \
#   --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
#   --do_train \
#   --output_dir=bert_model_output

# Disable for kernel

if 0:

    flags.DEFINE_string(
        "bert_config_file", "/kaggle/input/bertjointbaseline/bert_config.json",
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    flags.DEFINE_string("vocab_file", "/kaggle/input/bertjointbaseline/vocab-nq.txt",
                        "The vocabulary file that the BERT model was trained on.")

    flags.DEFINE_string(
        "output_dir", "outdir",
        "The output directory where the model checkpoints will be written.")

    # CHANGED
    flags.DEFINE_string("train_precomputed_file", "/kaggle/input/bertjointbaseline/nq-train.tfrecords-00000-of-00001",
                        "Precomputed tf records for training.")
    # CHANGED
    flags.DEFINE_integer("train_num_precomputed", 494670,
                         "Number of precomputed tf records for training.")

    flags.DEFINE_string(
        "output_prediction_file", "predictions.json",
        "Where to print predictions in NQ prediction format, to be passed to"
        "natural_questions.nq_eval.")

    flags.DEFINE_string(
        "init_checkpoint", "/kaggle/input/bertjointbaseline/bert_joint.ckpt",
        "Initial checkpoint (usually from a pre-trained BERT model).")

    flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

    # CHANGED - originally 384
    flags.DEFINE_integer(
        "max_seq_length", 512,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

    flags.DEFINE_integer(
        "doc_stride", 128,
        "When splitting up a long document into chunks, how much stride to "
        "take between chunks.")

    flags.DEFINE_integer(
        "max_query_length", 64,
        "The maximum number of tokens for the question. Questions longer than "
        "this will be truncated to this length.")

    # CHANGED
    flags.DEFINE_bool("do_train", True, "Whether to run training.")

    # CHANGED
    flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

    flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

    flags.DEFINE_integer("predict_batch_size", 8,
                         "Total batch size for predictions.")

    # CHANGED - taken from the BERT on nq - can be tuned - original 5e-5
    flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

    # CHANGED - originally 3
    flags.DEFINE_float("num_train_epochs", 1.0,
                       "Total number of training epochs to perform.")

    flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

    # CHANGED - originally 1000
    flags.DEFINE_integer("save_checkpoints_steps", 5000,
                         "How often to save the model checkpoint.")

    flags.DEFINE_integer("iterations_per_loop", 1000,
                         "How many steps to make in each estimator call.")

    flags.DEFINE_integer(
        "n_best_size", 20,
        "The total number of n-best predictions to generate in the "
        "nbest_predictions.json output file.")

    flags.DEFINE_integer(
        "verbosity", 1, "How verbose our error messages should be")

    flags.DEFINE_integer(
        "max_answer_length", 30,
        "The maximum length of an answer that can be generated. This is needed "
        "because the start and end predictions are not conditioned on one another.")

    flags.DEFINE_float(
        "include_unknowns", -1.0,
        "If positive, probability of including answers of type `UNKNOWN`.")

    flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
    flags.DEFINE_bool("use_one_hot_embeddings", False, "Whether to use use_one_hot_embeddings")

    absl.flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

    flags.DEFINE_bool(
        "verbose_logging", False,
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal NQ evaluation.")

    flags.DEFINE_boolean(
        "skip_nested_contexts", True,
        "Completely ignore context that are not top level nodes in the page.")

    flags.DEFINE_integer("task_id", 0,
                         "Train and dev shard to read from and write to.")

    flags.DEFINE_integer("max_contexts", 48,
                         "Maximum number of contexts to output for an example.")

    flags.DEFINE_integer(
        "max_position", 50,
        "Maximum context position for which to generate special tokens.")


    ## Special flags - do not change

    flags.DEFINE_string(
        "predict_file", "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl",
        "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
    flags.DEFINE_boolean("logtostderr", True, "Logs to stderr")
    flags.DEFINE_boolean("undefok", True, "it's okay to be undefined")
    flags.DEFINE_string('f', '', 'kernel')
    flags.DEFINE_string('HistoryManager.hist_file', '', 'kernel')

    FLAGS = flags.FLAGS
    FLAGS(sys.argv) # Parse the flags



# Code for training
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