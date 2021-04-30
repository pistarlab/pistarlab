import tensorflow as tf; 
print(tf.__version__)
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(tf.config.list_physical_devices('GPU'))
