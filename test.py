import tensorflow as tf
import sys

tensor = tf.range(10)
tf.print(tensor, output_stream=sys.stderr)
print(tf.shape(tensor, out_type=tf.dtypes.int32, name=None))