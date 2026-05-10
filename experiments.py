from preparation import prepare_data
from perceptron import Perceptron

from sklearn import metrics

import matplotlib.pyplot as plt

def train_and_evaluate(lr=0.1, batch_size=32, ep = 100, init_type="small_random"):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    model = Perceptron(
        2,
        init_type=init_type
    )

    train_loss, val_loss = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=ep,
        lr=lr,
        batch_size=batch_size
    )

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = metrics.accuracy_score(
        y_train,
        pred_train
    )

    test_acc = metrics.accuracy_score(
        y_test,
        pred_test
    )

    return train_loss, val_loss, train_acc, test_acc

learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]

batch_size = 32

for lr in learning_rates:

    train_loss, val_loss, train_acc, test_acc = (
        train_and_evaluate(
            lr=lr,
            batch_size=32
        )
    )

    plt.plot(
        val_loss,
        label=f"lr={lr}"
    )

    print(
        f"lr={lr} | "
        f"train_acc={train_acc:.3f} | "
        f"test_acc={test_acc:.3f}"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Rate Experiment")
plt.legend()
plt.show()

batch_sizes = [1, 16, 32, 64, 256]
lr = 0.1

plt.figure(figsize=(10, 6))

for bs in batch_sizes:

    train_loss, val_loss, train_acc, test_acc = (
        train_and_evaluate(
            lr=0.1,
            batch_size=bs
        )
    )

    plt.plot(
        val_loss,
        label=f"batch_size={bs}"
    )

    print(
        f"batch_size={bs} | "
        f"train_acc={train_acc:.3f} | "
        f"test_acc={test_acc:.3f}"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Batch Size Experiment")
plt.legend()
plt.show()

init_types = [
    "zeros",
    "small_random",
    "large_random"
]

plt.figure(figsize=(10, 6))

for init_type in init_types:

    train_loss, val_loss, train_acc, test_acc = (
        train_and_evaluate(
            init_type=init_type
        )
    )

    plt.plot(
        val_loss,
        label=init_type
    )

    print(
        f"{init_type} | "
        f"train_acc={train_acc:.3f} | "
        f"test_acc={test_acc:.3f}"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Weight Initialization Experiment")

plt.legend()

plt.show()