import tensorflow as tf
import numpy as np

def create_model():
    """Create CNN model without MPI"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
    x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0

    # Create and train model
    model = create_model()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    print("Training...")
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('mnist_cnn.h5')
    print("Model saved as mnist_cnn.h5")

if __name__ == "__main__":
    main()