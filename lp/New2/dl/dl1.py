import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load and inspect data
df = pd.read_csv('BostonHousing.csv')

# Handle missing values
df = df.dropna()  # or use imputation

# Prepare data
X = df.drop('MEDV', axis=1).values
y = df['MEDV'].values.reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=45
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model with improvements
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='linear')
])

# Safer optimizer configuration
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,  # Explicit batch size
    validation_split=0.05,
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, y_test, verbose=0)
r2 = r2_score(y_test, y_pred)

print('\nEvaluation metrics:')
print(f'Mean squared error: {mse_nn:.4f}')
print(f'Mean absolute error: {mae_nn:.4f}')
print(f'R-squared score: {r2:.4f}')

# Plot results
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, c='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], 
        [y_test.min(), y_test.max()], 
        'b--', lw=2)
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title('True vs Predicted Values', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()