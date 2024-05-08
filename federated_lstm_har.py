import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random as python_random

# The below is necessary for starting core Python generated random numbers in a well-defined state.
python_random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Model definition
def create_model(sequence_length=40, num_joints=17, num_classes=5):
    return Sequential([
        LSTM(256, input_shape=(sequence_length, num_joints*3), return_sequences=True),
        # BatchNormalization(),
        Dropout(0.2), # Drop out for regularization
        LSTM(128, return_sequences=True),
        # BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(num_classes, activation='softmax')
    ])

def split_data(X,y,ratios):
    assert np.sum(ratios) == 1, "The sum of the ratios must be 1"
    total_size = X.shape[0]
    print('total size:', total_size)
    splits = np.cumsum([int(total_size*r) for r in ratios])
    print('splits:', splits)
    return np.split(X, splits[:-1]), np.split(y, splits[:-1])

# Federated data preparation
def preprocess(dataset):
    return dataset.shuffle(1196).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

def make_federated_data(X, y, num_clients=4):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    client_datasets = [preprocess(dataset) for _ in range(num_clients)]
    return client_datasets

def make_federated_data_ratio(X, y, ratios):
    X_splits, y_splits = split_data(X,y,ratios)
    client_datasets = [preprocess(tf.data.Dataset.from_tensor_slices((x,y))) for x,y in zip(X_splits,y_splits)]
    return client_datasets

# Federated training process definition
def model_fn():
    keras_model = create_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 40, 51], dtype=tf.float64), tf.TensorSpec(shape=[None, 5], dtype=tf.float32)),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

# Evaluation function
def evaluate_model(model, test_data, test_labels):
    y_pred_probs = model.predict(test_data)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred, average="macro"):.4f}')
    print(f'Recall: {recall_score(y_true, y_pred, average="macro"):.4f}')
    print(f'F1-Score: {f1_score(y_true, y_pred, average="macro"):.4f}')

def evaluate_global_model(keras_model, X_test, y_test):
    loss, accuracy = keras_model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy


# Main script
tff.backends.native.set_local_execution_context()



# load the data
sequence_length = 40
data_2d = np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/simplified_combined_data_reshaped.csv',delimiter=',')
labels = np.loadtxt('/home/bingyi/Desktop/NCSU/Research/federated_lstm/combined_labels.csv',delimiter=',')
data = data_2d.reshape(data_2d.shape[0],sequence_length,-1)
data = np.nan_to_num(data)
one_hot_labels = to_categorical(labels, num_classes=5)
print("data shape: ", data.shape)
print("label shape: ", one_hot_labels.shape)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, one_hot_labels,test_size=0.2, random_state=42)


# X_train, y_train = generate_synthetic_data(num_samples=1000)
# print(X_train.shape)
# X_test, y_test = generate_synthetic_data(num_samples=200)
federated_train_data = make_federated_data(X_train, y_train)
# federated_train_data = make_federated_data_ratio(X_train, y_train, ratios=[0.2,0.3,0.5])
federated_train_data = make_federated_data_ratio(X_train, y_train, ratios=[0.2, 0.2, 0.2, 0.2, 0.2])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02), # 0.02 works well
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.005),
)
state = iterative_process.initialize()

NUM_ROUNDS = 100
best_loss = float('inf')
patience_counter = 0
patience = 10
for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f'Round {round_num}, Metrics: {metrics}')
    keras_model = create_model()
    state.model.assign_weights_to(keras_model)
    keras_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    current_loss, _ = evaluate_global_model(keras_model, X_test, y_test)
    # print(f'Round {round_num}, Metrics: {metrics}')

    # early stopping 
    if current_loss < best_loss - 1e-4:
        best_state = state
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Stopping early at round {round_num}")
        break

# Extract and evaluate the global model
keras_model = create_model()
best_state.model.assign_weights_to(keras_model)
evaluate_model(keras_model, X_test, y_test)


y_pred_probs = keras_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
# Assuming y_true_classes and y_pred_classes are defined from your model's predictions
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()