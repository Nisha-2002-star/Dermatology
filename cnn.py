import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

batch_size = 32

# Data augmentation and rescaling
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'DataSet/train', target_size=(200, 200), batch_size=batch_size,
    classes=['Acne','dry', 'normal', 'oily'], class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'DataSet/test', target_size=(200, 200), batch_size=batch_size,
    classes=['Acne','dry', 'normal', 'oily'], class_mode='categorical', shuffle=False)

# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

total_sample = train_generator.n
n_epochs = 15

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=int(total_sample / batch_size),
    epochs=n_epochs,
    callbacks=[early_stopping],
    verbose=1
)

model.save('model.h5')

# Plot training history
acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Model evaluation
print('Generating classification report and confusion matrix...')
test_steps_per_epoch = math.ceil(test_generator.samples / test_generator.batch_size)

predictions = model.predict(test_generator, steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print('Classification Report')
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print('Confusion Matrix')
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
