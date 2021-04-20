import io
import os
import re
import shutil
import string
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

batch_size = 1024
seed = 123
#train_ds = tf.keras.preprocessing.text_dataset_from_directory('/Users/sairo/Documents/ark/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
