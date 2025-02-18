
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

# Load your dataset
df2 = pd.read_csv('modified_data.csv')

# Define input features (X) and target variables (Y)
X = df2[['Age', 'Gender', 'Weight (kg)', 'Height (m)']].values  # Inputs
Y = df2[['BMI', 'Fat_Percentage', 'BMR', 'LBM', 'SMM']].values  # Outputs

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# Define the neural network model
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(Y_train_scaled.shape[1])  # Output layer for 5 target variables
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Add Early Stopping for better performance
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled, Y_train_scaled,
    validation_data=(X_test_scaled, Y_test_scaled),
    epochs=200,
    batch_size=16,  # Smaller batch size for frequent updates
    callbacks=[early_stopping],
    verbose=2
)

# Save the trained model and scalers
model.save('body_metrics_model.keras')

with open('body_metrics_scalers.pkl', 'wb') as scaler_file:
    pickle.dump({'scaler_X': scaler_X, 'scaler_Y': scaler_Y}, scaler_file)

# Evaluate the model
loss, mae = model.evaluate(X_test_scaled, Y_test_scaled, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
