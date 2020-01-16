from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os

import numpy as np
import IPython.display as display
import json

# Load one example
os.chdir('C:/Users/diman/Documents/Kaggle 2019 Data/tensorflow2')

with open('simplified-nq-train_one.jsonl') as f:
    line = f.readline()
    example_data = json.loads(line)[0]

example_data.keys()
example_data['document_text']
example_data['long_answer_candidates']
example_data['question_text']
example_data['annotations']
example_data['document_url']
example_data['example_id']



os.chdir('C:/Users/diman/OneDrive/Work_temp/Insight/Kaggle Competition Insight Team/tensorflow2-question-answering')

tfrecord_location = 'C:/Users/diman/Documents/Kaggle 2019 Data/tensorflow2/Others work/bert-joint-baseline'
name = "nq-train.tfrecords"
filename = os.path.join(tfrecord_location, name)

dataset = tf.data.TFRecordDataset(filename)




for raw_record in dataset.take(1):
  print(repr(raw_record))
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)
  print(type(example))
  

# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)







def decode(serialized_example):
  """
  Parses an image and label from the given `serialized_example`.
  It is used as a map function for `dataset.map`
  """
  IMAGE_SIZE = 28
  IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
  
  # 1. define a parser
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # 2. Convert the data
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int32)
  # 3. reshape
  image.set_shape((IMAGE_PIXELS))
  return image, label

dataset = dataset.map(decode)

