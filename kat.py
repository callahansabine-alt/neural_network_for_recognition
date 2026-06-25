import tensorflow as tf
from keras.src.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random

model = tf.keras.models.load_model("kam.h5")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test/255.0
x_test = x_test.reshape(x_test.shape -1, 28 * 28)

index = random.randint(0, len(x_test))
random_picture = x_test[index]
random_picture = random_picture.reshape(1, 28 * 28)

prediction = model.predict(random_picture)
predicted_label = prediction.argmax()
true_label = to_categorical(y_test, num_classes=10)[index].argmax()

plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
plt.title(f"Predicted label: {predicted_label}, Actual label: {true_label}")


plt.axis("off")
plt.show()