
# Import tensorflow and helper libraries
import tensorflow as tf
import os
import re
import shutil
import string

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.__version__)

# Next, import a dataset
# containing both positive and negative movie reviews

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

#dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
#dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
dataset = os.path.join('aclImdb_v1/')

dataset_dir = os.path.join(dataset, 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')

os.listdir(train_dir)

# This needs to be run only once after the data is downloaded
# Remove unwanted folder from training data directory
"""
removable = os.path.join(train_dir, 'unsup')
shutil.rmtree(removable)

"""

# Create raw dataset from downloaded text files
batch_size = 32
seed = 42

# This creates a dataset for training
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

# Display sample data
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

# Check label classes
print('0:', raw_train_ds.class_names[0])
print('1:', raw_train_ds.class_names[1])

# Now create a dataset for validation
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed,
)

# Now create the test dataset
test_dir = os.path.join(dataset_dir, 'test')

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size,
)

# Now prepare the dataset for training the model
# For this, conver the text into 'standard' form

# This function removes HTML from input text
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
            '[%s]' % re.escape(string.punctuation),
            '')


# Now add a neural network layer to vectorize data
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset without labels
# to vectorize training data
train_text = raw_train_ds.map(lambda x, y: x)

vectorize_layer.adapt(train_text)

# Test the vectorization layer
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Retrieve some data from training set
# for testing vectorization layer
text_batch, label_batch = next(iter(raw_train_ds))

# Each _batch has 32 values
first_review, first_label = text_batch[0], label_batch[0]

print("Review:", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized text:", vectorize_text(first_review, first_label))

print('26 --->', vectorize_layer.get_vocabulary()[26])
print('13 ---> ', vectorize_layer.get_vocabulary()[13])
print("Vocabulary size: {}" .format(len(vectorize_layer.get_vocabulary())))

# Create datasets for training model

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Configure datasets for performance

AUTOTUNE = tf.data.AUTOTUNE

train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])

# Finally, train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
 
loss, accuracy = model.evaluate(test_ds)

print(loss)
print(accuracy)
 
history_dict = history.history
history_dict.keys()
 
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
 
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']
)

# Test export_model on raw_strings
metrics = export_model.evaluate(raw_test_ds, return_dict=True)
print(metrics)

# Test export_model on sample data
examples = tf.constant([
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
])

export_model.predict(examples)


