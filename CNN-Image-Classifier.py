# Cat-Dog Dataset Code

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf

# Class names for the dataset
class_names = ['cat', 'dog']

# Parameters for testing
img_sizes = [32, 64, 112]  # Image sizes to test
conv_layer_counts = [1, 2, 3]  # Number of convolutional layers to test
num_epochs = 15
batch_size = 160

# Function to normalize pixel values
def normalize_images(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # normalize pixel values to [0, 1] instead of 255
    return image, label

# Function to create the model
def create_model(img_size, conv_layer_count):
    model = keras.models.Sequential()
    for i in range(conv_layer_count):
        if i == 0:  # First convolutional layer
            model.add(keras.layers.Conv2D(
                filters=32 * (i + 1),  # increasing filters with each layer
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                input_shape=(img_size, img_size, 3)  # Define input shape for the first layer
            ))
        else:  # Subsequent convolutional layers
            model.add(keras.layers.Conv2D(
                filters=32 * (i + 1),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            ))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(units=1))  # this is output layer for binary classification, either cat or not cat in this case i think

    return model

# Placeholder for results
results = []

# Iterate through combinations of image sizes and convolutional layers
for conv_layer_count in conv_layer_counts:
    for img_size in img_sizes:
        print(f"Training with {conv_layer_count} conv layers and img_size={img_size}")

        # Load the dataset
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            r'\Cat-Dog\train',
            labels='inferred',
            label_mode='binary',  # Binary labels (0 or 1 for cat or dog)
            image_size=(img_size, img_size),
            batch_size=batch_size
        )
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            r'\Cat-Dog\test',
            labels='inferred',
            label_mode='binary',  # Binary labels (0 or 1 for cat or dog)
            image_size=(img_size, img_size),
            batch_size=batch_size
        )

        # Normalize the datasets
        train_dataset = train_dataset.map(normalize_images)
        test_dataset = test_dataset.map(normalize_images)

        # optimimzing for performance
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Create and compile the model
        model = create_model(img_size, conv_layer_count)
        model.compile(optimizer='adam',
                      loss=keras.losses.BinaryCrossentropy(from_logits=True), #using binary
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=test_dataset,
            verbose=1
        )

        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
        print(f"Test accuracy: {test_acc}")

        # Save results
        results.append({
            'img_size': img_size,
            'conv_layers': conv_layer_count,
            'test_accuracy': test_acc,
            'history': history.history
        })

# Plot results for each combination
for result in results:
    plt.figure(figsize=(10, 6))
    plt.plot(result['history']['accuracy'], label='Train Accuracy')
    plt.plot(result['history']['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Conv Layers: {result['conv_layers']}, Image Size: {result['img_size']}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()

#----------------------------------------------------------------
# Visual Domain Decathlon Code

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf

# Class names for the dataset
class_names = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']

# Parameters for testing
img_sizes = [32, 64, 112]  # Image sizes to test
conv_layer_counts = [1, 2, 3]  # Number of convolutional layers to test
num_epochs = 15
batch_size = 160

# Preprocessing function for normalization
def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0  # Normalize images to range [0, 1]
    return images, labels

# Function to create the model
def create_model(img_size, conv_layer_count):
    model = keras.models.Sequential()
    for i in range(conv_layer_count):
        if i == 0:  # First convolutional layer
            model.add(keras.layers.Conv2D(
                filters=32 * (i + 1),  # Increasing filters with each layer
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                input_shape=(img_size, img_size, 3)  # Define input shape for the first layer
            ))
        else:  # Subsequent convolutional layers
            model.add(keras.layers.Conv2D(
                filters=32 * (i + 1),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            ))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(units=10))

    return model

# Placeholder for results
results = []

# Iterate through combinations of image sizes and convolutional layers
for conv_layer_count in conv_layer_counts:
    for img_size in img_sizes:
        print(f"Training with {conv_layer_count} conv layers and img_size={img_size}")

        # Load the dataset
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            r'\VisualDecathlon_Dataset\train',
            labels='inferred',
            label_mode='int',
            image_size=(img_size, img_size),
            batch_size=batch_size
        )
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            r'\VisualDecathlon_Dataset\test',
            labels='inferred',
            label_mode='int',
            image_size=(img_size, img_size),
            batch_size=batch_size
        )

        # Normalize pixel values
        train_dataset = train_dataset.map(normalize_images)
        test_dataset = test_dataset.map(normalize_images)

        # optimimzing for performance
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Create and compile the model
        model = create_model(img_size, conv_layer_count)
        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # not binary here
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=test_dataset,
            verbose=1
        )

        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
        print(f"Test accuracy: {test_acc}")

        # Save results
        results.append({
            'img_size': img_size,
            'conv_layers': conv_layer_count,
            'test_accuracy': test_acc,
            'history': history.history
        })

# Plot results for each combination
for result in results:
    plt.figure(figsize=(10, 6))
    plt.plot(result['history']['accuracy'], label='Train Accuracy')
    plt.plot(result['history']['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Conv Layers: {result['conv_layers']}, Image Size: {result['img_size']}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()
