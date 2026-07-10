import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load MNIST Dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training Images:", x_train.shape)
print("Testing Images:", x_test.shape)

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
print("\nTraining Model...\n")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy * 100, "%")

# -----------------------------
# Save Model
# -----------------------------
model.save("digit_recognizer_model.h5")

print("\nModel Saved Successfully!")

# -----------------------------
# Predict Random Test Image
# -----------------------------
index = np.random.randint(0, len(x_test))

prediction = model.predict(x_test[index].reshape(1,28,28,1))

predicted_digit = np.argmax(prediction)
actual_digit = y_test[index]

print("\nPredicted Digit:", predicted_digit)
print("Actual Digit:", actual_digit)

# -----------------------------
# Display Image
# -----------------------------
plt.imshow(x_test[index].reshape(28,28), cmap="gray")
plt.title(f"Predicted: {predicted_digit} | Actual: {actual_digit}")
plt.axis("off")
plt.show()
