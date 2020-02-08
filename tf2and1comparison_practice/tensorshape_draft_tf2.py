# This is tf 2

import tensorflow as tf

print(tf.__version__)

# Create a shape and choose an index

i = 0
shape = tf.TensorShape([16, None, 256])
shape

# If you had this in TF 1.x:

value = shape[i].value

# Then do this in TF 2.0:

value = shape[i]
value

# If you had this in TF 1.x:

for dim in shape:
    value = dim.value
    print(value)

# Then do this in TF 2.0:

for value in shape:
  print(value)
# 
# If you had this in TF 1.x (Or used any other dimension method):

other_dim = 16

dim = shape[i]
dim.assert_is_compatible_with(other_dim)
# 
# Then do this in TF 2.0:


Dimension = tf.compat.v1.Dimension

if shape.rank is None:
  dim = Dimension(None)
else:
  dim = shape.dims[i]
dim.is_compatible_with(other_dim) # or any other dimension method


# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
type(mnist_images)
mnist_images.shape
mnist_images[0]
mnist_labels
mnist_labels[0]

mnist_images = mnist_images/255

dataset = tf.data.Dataset.from_tensor_slices((mnist_images, mnist_labels))
dataset


# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

mnist_images[...,tf.newaxis].shape
mnist_images.shape

# Build the model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])

count = 0
for images,labels in dataset.take(1):
    print(count)
    count += 1
    print(images.shape)
    print("Logits: ", mnist_model(images[0:1]).numpy())


dtype = tf.as_dtype(images.dtype or tf.keras.backend.floatx())

input_shape = tf.TensorShape([4,None])

print(tf.compat.dimension_value(input_shape[-1]))
print(input_shape[-1])

[a for a in [0,1,2,None] if a is not None]