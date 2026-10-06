# Binary sentiment classification on the IMDB movie-review corpus.
# Diagnostic prints, plotting and model export are omitted

import os, re, string
import tensorflow as tf
from tensorflow.keras import layers, losses

batch_size, seed = 32, 42
max_features, sequence_length, embedding_dim = 10000, 250, 16

dataset_dir = os.path.join('aclImdb_v1/', 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# 1. Load the labelled text files as three datasets.
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir, batch_size=batch_size)

# 2. Standardise and vectorise: lower-case, strip HTML and punctuation,
#    then map each review to a fixed-length vector of token indices.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped, '[%s]' % re.escape(string.punctuation), '')

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization, max_tokens=max_features,
    output_mode='int', output_sequence_length=sequence_length)
vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

def vectorize_text(text, label):
    return vectorize_layer(tf.expand_dims(text, -1)), label

train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(tf.data.AUTOTUNE)
val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(tf.data.AUTOTUNE)
test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(tf.data.AUTOTUNE)

# 3. The model: a learned embedding, averaged over the review, then one
#    sigmoid output unit.
model = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')])

model.compile(loss=losses.BinaryCrossentropy(), optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])

# 4. Train and evaluate on held-out data.
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
loss, accuracy = model.evaluate(test_ds)
print(f"test loss {loss:.4f}, test accuracy {accuracy:.4f}")
