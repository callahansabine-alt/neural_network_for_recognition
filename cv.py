from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_images[i], cmap="gray")
    plt.title(f"Number {train_labels[i]}")
    plt.axis('off')
plt.show()

x_train = train_images/255
x_test = test_images/255

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)
print(train_labels)

model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(784,)),
        Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss = "categorical_crossentropy",
    metrics=["accuracy"])
kit = model.fit(
    x_train, train_labels,
    epochs=5, batch_size=32,
    validation_data=(x_test, test_labels)
)
model.save("kam.h5")