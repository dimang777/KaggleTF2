# This spyder is tf 1
import tensorflow as tf

print(tf.__version__)

# Create a shape and choose an index

i = 0
shape = tf.TensorShape([16, None, 256])
shape

# If you had this in TF 1.x:

value = shape[i].value
print(value)
# Then do this in TF 2.0:

value = shape[i]
value
print(value)
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
