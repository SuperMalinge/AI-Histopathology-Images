import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread as mpl_imread
from skimage.transform import resize
from tensorflow.keras.losses import MeanSquaredError

class ConLayerLeft(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerLeft, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = self.add_weight(shape=(kernel_size, kernel_size, in_channels, out_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))

    def call(self, inputs, stride=1):
        layer = tf.nn.conv2d(inputs, self.w, strides=[1, stride, stride, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA

class ConLayerRight(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerRight, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = self.add_weight(shape=(kernel_size, kernel_size, out_channels, in_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))

    def call(self, inputs, stride=1):
        current_shape_size = inputs.shape
        output_shape = [tf.shape(inputs)[0],
                       int(current_shape_size[1]),
                       int(current_shape_size[2]),
                       self.out_channels]
        
        layer = tf.nn.conv2d_transpose(inputs, self.w, output_shape=output_shape,
                                     strides=[1, 1, 1, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA

def normalize_histology(image):
    # Specialized normalization for H&E stained images
    return (image - np.mean(image)) / (np.std(image) + 1e-8)

# Data loading paths adjusted for histopathology
data_location = "./Histopathology/training/images/"
train_data = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():  # Common format for microscopy
            train_data.append(os.path.join(dirName, filename))

# Initialize with 3 channels for RGB histology images
train_images = np.zeros(shape=(128, 1024, 1024, 3))
train_labels = np.zeros(shape=(128, 1024, 1024, 1))

# Modified layer configurations for cellular features
layer_configs = [
    (64, 3), (64, 3),   # Fine cellular details
    (128, 3), (128, 3), # Tissue patterns
    (256, 3), (256, 3), # Complex structures
    (128, 3), (64, 3),  # Feature synthesis
    (32, 3), (1, 3)     # Output layer
]

class HistopathologyCNN(tf.keras.Model):
    def __init__(self, layers):
        super(HistopathologyCNN, self).__init__()
        self.layer_list = layers
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
            x = self.batch_norm(x)
        return x

# Training parameters optimized for histopathology
num_epochs = 150
init_lr = 0.00005
batch_size = 2

# Initialize model and compile
current_channels = 3  # Start with 3 channels for RGB
layers = []
for out_channels, kernel_size in layer_configs:
    layers.append(ConLayerLeft(kernel_size, current_channels, out_channels))
    current_channels = out_channels

model = HistopathologyCNN(layers)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
             loss='mse',
             metrics=['accuracy'])

# Training loop with visualization
for iter in range(num_epochs):
    for current_batch_index in range(0, len(train_images), batch_size):
        current_batch = train_images[current_batch_index:current_batch_index + batch_size]
        current_label = train_labels[current_batch_index:current_batch_index + batch_size]
        
        with tf.GradientTape() as tape:
            predictions = model(current_batch)
            loss_value = tf.reduce_mean(tf.square(predictions - current_label))
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if iter % 10 == 0:
            visualize_results(current_batch[0], current_label[0], predictions[0], iter)

def visualize_results(original, ground_truth, prediction, iter):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Original H&E Image')
    
    plt.subplot(132)
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.title('Ground Truth')
    
    plt.subplot(133)
    plt.imshow(prediction[:,:,0], cmap='jet')
    plt.title('Detected Cells')
    
    plt.savefig(f'results/iteration_{iter}.png')
    plt.close()

# Save the trained model
model.save('histopathology_model')
