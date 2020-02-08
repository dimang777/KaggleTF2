# %% [code] {"jupyter":{"outputs_hidden":true}}
# pre-processing
# tqml removed since it's not necessary. tqml.tqml_notebook(a) is the same as a
if 1:

    eval_records = "/content/bert-joint-baseline/nq-test.tfrecords"
    #nq_test_file = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'
    if not os.path.exists(eval_records):
        # tf2baseline.FLAGS.max_seq_length = 512
        
        # import tensorflow as tf
        # import os
        # eval_records = "C:/Users/diman/OneDrive/Work_temp/Insight/Kaggle Competition Insight Team/kaggleandoriginal_prediction_comparison/nq-test.tfrecords"
        # tf.io.TFRecordWriter(os.path.join(eval_records))
        
        
        eval_writer = tf1nq.FeatureWriter(
            filename=os.path.join(eval_records),
            is_training=False)
    
        tokenizer = tokenization.FullTokenizer(vocab_file='../input/bert-joint-baseline/vocab-nq.txt', 
                                               do_lower_case=True)
    
        features = []
    ################## bert_utils###################################################
    ################## bert_utils###################################################
    ################## bert_utils###################################################
        # Original - class - it's actually a function. Find convert below
        # convert = bert_utils.ConvertExamples2Features(tokenizer=tokenizer,
        #                                                is_training=False,
        #                                                output_fn=eval_writer.process_feature,
        #                                                collect_stat=False)
        # Changed to - look below where convert function used to be - a few lines below
        # tf1nq.convert_eamples_to_features(example, tokenizer=tokenizer,
        #                                                is_training=False,
        #                                                output_fn=eval_writer.process_feature)
    
        n_examples = 0
    ################## bert_utils###################################################
    ################## bert_utils###################################################
    ################## bert_utils###################################################
        # Original
        # for examples in bert_utils.nq_examples_iter(input_file=nq_test_file, 
        #                                        is_training=False,
        #                                        tqdm=tqdm_notebook):
        # Changed
        examples_iter = tf1nq.read_nq_examples(input_file=nq_test_file, is_training=False))
        for examples in examples_iter:
    
            for example in examples:
                # Original
                # n_examples += convert(example)
                # Changed to
                n_examples += tf1nq.convert_examples_to_features(example, tokenizer=tokenizer,
                                                                 is_training=False,
                                                                 output_fn=eval_writer.process_feature)
    
        eval_writer.close()
        print('number of test examples: %d, written to file: %d' % (n_examples,eval_writer.num_features))
