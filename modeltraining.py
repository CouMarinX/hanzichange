import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.models import Model

# Data Preprocessing Function
def preprocess_data(input_images, target_images):
    """
    Preprocess input and target images for training.
    """
    input_images = input_images / 255.0  # Normalize to [0, 1]
    target_images = target_images / 255.0  # Normalize to [0, 1]
    return input_images, target_images

# Load Dataset Function
def load_dataset(input_dir, target_dir, image_size=(128, 128)):
    
    input_images = []
    target_images = []

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(input_path) and os.path.exists(target_path):
            # Load and resize input image
            input_img = cv2.imread(input_path)
            input_img = cv2.resize(input_img, image_size)
            input_images.append(input_img)

            # Load and resize target image
            target_img = cv2.imread(target_path)
            target_img = cv2.resize(target_img, image_size)
            target_images.append(target_img)

    # Convert to NumPy arrays and normalize to [0, 1]
    input_images = np.array(input_images) / 255.0
    target_images = np.array(target_images) / 255.0

    return input_images, target_images

# W-Net Architecture
def build_w_net(input_shape):
    """
    Build a W-Net for stylized Chinese character generation.
    """
    def build_u_net(input_layer):
        # Encoding path
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        
        # Decoding path
        up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
        up1 = concatenate([up1, conv2])
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
        up2 = concatenate([up2, conv1])
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

        output_layer = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
        return output_layer

    # Input and W-Net layers
    input_layer = Input(shape=input_shape)
    first_u_net_output = build_u_net(input_layer)
    second_u_net_output = build_u_net(first_u_net_output)

    return Model(inputs=input_layer, outputs=second_u_net_output)

# Compile and Train the Model
def train_w_net(model, input_images, target_images, epochs=50, batch_size=16):
    """
    Compile and train the W-Net model.
    """
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(
        input_images, target_images,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )
    return history

# Visualization Function
def visualize_results(model, test_images, target_images):
    """
    Visualize test results.
    """
    predictions = model.predict(test_images)

    for i in range(3):
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.imshow(test_images[i])

        plt.subplot(1, 3, 2)
        plt.title("Generated")
        plt.imshow(predictions[i])

        plt.subplot(1, 3, 3)
        plt.title("Target")
        plt.imshow(target_images[i])

        plt.show()

# Main Program
if __name__ == "__main__":
    # Placeholder: Load your dataset of Chinese characters and their stylized versions.
    input_images = np.random.rand(100, 128, 128, 3)  # Example placeholder data
    target_images = np.random.rand(100, 128, 128, 3)  # Example placeholder data

    # Preprocess the data
    input_images, target_images = preprocess_data(input_images, target_images)

    # Build the W-Net model
    input_shape = input_images.shape[1:]
    w_net_model = build_w_net(input_shape)
    
    input_dir = "dataset/inputs"  # 输入汉字图片的文件夹路径
    target_dir = "dataset/targets"  # 风格化目标图片的文件夹路径

    # 加载数据集
    input_images, target_images = load_dataset(input_dir, target_dir)
    
    # Train the model
    history = train_w_net(w_net_model, input_images, target_images)

    # Visualize results
    test_images = input_images[:10]
    test_targets = target_images[:10]
    visualize_results(w_net_model, test_images, test_targets)
