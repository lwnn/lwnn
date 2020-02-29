import sys
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio
from tensorflow.python.framework import graph_util
import numpy as np
# code from https://github.com/mozilla/DeepSpeech

def samples_to_mfccs(samples, sample_rate):

    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=512,
                                                  stride=320,
                                                  magnitude_squared=True)

    mfccs = contrib_audio.mfcc(spectrogram=spectrogram,
                               sample_rate=sample_rate,
                               dct_coefficient_count=26,
                               upper_frequency_limit=4000)
    mfccs = tf.reshape(mfccs, [-1, 26])

    return mfccs, tf.shape(input=mfccs)[0]


def audiofile_to_features():
    samples = tf.placeholder(tf.string, name='input')
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate)

    return features, features_len


def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * 9 + 1
    num_channels = 26

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    filter = np.eye(window_width * num_channels).reshape(window_width, num_channels, window_width * num_channels)
    eye_filter = tf.constant(filter, tf.float32) # pylint: disable=bad-continuation
    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels], name='output')

    return batch_x

with tf.Session() as sess:
    features, _ = audiofile_to_features()
    features = tf.expand_dims(features, 0)
    features = create_overlapping_windows(features)
    if(len(sys.argv) > 1):
        wav_data = open(sys.argv[1], 'rb').read()
        features_o = sess.run(features, {'input:0': wav_data})
        print(features_o.shape)
        features_o.tofile('input.raw')
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
    with tf.gfile.FastGFile('./mfcc.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
