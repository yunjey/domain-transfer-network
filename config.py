import tensorflow as tf

# configuration for tensorflow 0.11 and 0.12 version
try:
    # tensorflow 0.12 version
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge_all
    SummaryWriter = tf.summary.FileWriter
except:
    # tensorflow <= 0.11 version
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_all_summaries
    SummaryWriter = tf.train.SummaryWriter