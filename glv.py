from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'C:/Users/sairo/Documents/ark/data1/1_3-1_9 COVID-19 Long-Haulers Discussion Group.docx_Posts.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)


# ---------------------
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
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'C:/Users/sairo/Documents/ark/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'C:/Users/sairo/Documents/ark/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

result = embedding_layer(tf.constant([1,2,3]))
result.numpy()
# result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
# result.shape
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])

model.summary()

weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()


out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if  index == 0: continue # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()


