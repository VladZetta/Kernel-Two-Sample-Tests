# preprocess_mnist.py

from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    # Fetch the MNIST dataset and explicitly set the parser if needed.
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    # Convert to NumPy arrays and ensure correct types.
    data = mnist['data'].astype('float32').to_numpy() if hasattr(mnist['data'], 'to_numpy') else np.array(mnist['data'], dtype='float32')
    labels = np.array(mnist['target']).astype(int)
    
    # Normalize pixel values to [0, 1]
    data /= 255.0
    
    return data, labels

if __name__ == '__main__':
    data, labels = load_mnist()
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
