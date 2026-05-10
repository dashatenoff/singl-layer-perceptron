from preparation import prepare_data
from perceptron import Perceptron

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

model = Perceptron(2)

train_loss, test_loss = model.fit(X_train, y_train, X_val, y_val, 100, 0.1, 32)

pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test, pred)

pred_train = model.predict(X_train)
train_acc = metrics.accuracy_score(y_train, pred_train)

print(train_acc)
print(acc)

plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap='bwr',
    alpha=0.7
)
x_min = X_test[:, 0].min()
x_max = X_test[:, 0].max()
x_line = np.linspace(x_min, x_max, 100)
w1 = model.w[0]
w2 = model.w[1]
b = model.b
y_line = -(w1 * x_line + b) / w2

plt.plot(x_line, y_line, color='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()