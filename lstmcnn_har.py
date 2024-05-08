import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
import random as python_random

# The below is necessary for starting core Python generated random numbers in a well-defined state.
python_random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def create_model():

    # define the LSTM model
    model = Sequential([
        # LSTM layer with 50 units 
        LSTM(256, input_shape=(40,51), return_sequences=True),
        # BatchNormalization(),
        Dropout(0.2), # Drop out for regularization
        LSTM(128, return_sequences=True),
        # BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(5,activation='softmax')
    ])   
    return model 

def create_cnn_lstm_model(sequence_length=40, num_joints=17, num_classes=5):
    model = Sequential([
        # Convolutional layer learns local patterns
        Conv1D(64, 3, activation='relu', input_shape=(sequence_length, num_joints*3)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        # LSTM layer learns long-term dependencies
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        # Dense layer for output
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_lstm_cnn_model(sequence_length=40, num_joints=17, num_classes=5):
    return Sequential([
        # First LSTM layer
        LSTM(256, return_sequences=True, input_shape=(sequence_length, num_joints*3)),
        Dropout(0.2),
        
        # Second LSTM layer, make sure to return sequences
        LSTM(128, return_sequences=True),
        Dropout(0.2),

        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        # Flatten(),
        # Flatten to prepare for the dense layer
        Flatten(),
        
        # Dense layer for classification
        Dense(num_classes, activation='softmax')
    ])

def split_data_into_clients(X, y, proportions):
    if sum(proportions) != 1:
        raise ValueError("Proportions must sum to 1")
    
    client_data = []
    start_idx = 0
    n_samples = len(X)
    
    for proportion in proportions:
        end_idx = start_idx + int(proportion * n_samples)
        X_client = X[start_idx:end_idx]
        y_client = y[start_idx:end_idx]
        client_data.append((X_client, y_client))
        start_idx = end_idx
    
    return client_data


def train_models_on_clients(client_data, X_test,y_test):
    models = []
    histories = []
    for X_client, y_client in client_data:
        print(X_client.shape)
        print(y_client.shape)
        model = create_lstm_cnn_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_client, y_client, epochs=1000,batch_size=32,validation_data=(X_test,y_test), callbacks=[early_stopping], verbose=0)
        evaluate_model(model, X_test, y_test)
        models.append(model)
        histories.append(history)
    return models, histories


# define the LSTM model
model = Sequential([
    # LSTM layer with 50 units 
    LSTM(256, input_shape=(40,51), return_sequences=True),
    # BatchNormalization(),
    Dropout(0.2), # Drop out for regularization
    LSTM(128, return_sequences=True),
    # BatchNormalization(),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(5,activation='softmax')
])

# Evaluation function
def evaluate_model(model, test_data, test_labels):
    y_pred_probs = model.predict(test_data)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred, average="macro"):.4f}')
    print(f'Recall: {recall_score(y_true, y_pred, average="macro"):.4f}')
    print(f'F1-Score: {f1_score(y_true, y_pred, average="macro"):.4f}')

early_stopping = EarlyStopping(
    monitor='val_loss',     # Monitor the validation loss
    min_delta=1e-4,         # The minimum amount of change to qualify as an improvement
    patience=10,            # How many epochs to wait before stopping when seeing no improvement
    verbose=1,              # To log the stopping event
    mode='auto',            # Infers the direction of monitoring (minimizing or maximizing)
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

# compile the model
model = create_lstm_cnn_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model summary to check the final model architecture
model.summary()


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

# Train the model
history = model.fit(X_train,y_train,epochs=40,batch_size=32,validation_data=(X_test,y_test), callbacks=[early_stopping], verbose=1)
# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis=1)
y_true_classes = np.argmax(y_test,axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true_classes,y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

evaluate_model(model, X_test, y_test)


# client models
client_data = split_data_into_clients(X_train, y_train, proportions = [0.2,0.3,0.5])
models, histories = train_models_on_clients(client_data, X_test,y_test)  # Adjust num_classes accordingly



# plot the training and validation loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Assuming y_true_classes and y_pred_classes are defined from your model's predictions
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()