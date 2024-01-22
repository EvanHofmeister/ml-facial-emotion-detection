# build_train_evaluate.py

# Import necessary modules
# (Make sure all these are installed in your environment)
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from build_train_model import build_advanced_model, train_model, setup_data_generators
from metrics import plot_model_performance, plot_confusion_matrix

def build_train_evaluate(config, input_shape, num_classes, train_generator, validation_generator, epochs, config_name):
    """
    Build, train, and evaluate a model based on the given configuration.
    """
    # Build the model
    model = build_advanced_model(config, input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = train_model(model, train_generator, validation_generator, epochs, config_name)

    # Plot model performance
    plot_model_performance(history)

    # Evaluate the model
    true_labels = validation_generator.classes
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Class names from the generator
    class_names = list(validation_generator.class_indices.keys())

    # Plotting the confusion matrix
    plot_confusion_matrix(true_labels, predicted_classes, class_names)

    return model

def main():
    # Define configuration, input_shape, num_classes, etc.
    input_shape = (48, 48, 1)
    num_classes = 7
    epochs = 15
    config_name = 'final'
    advanced_config = {
        'layers': [
            {'type': 'conv2d', 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same',
             'input_shape': input_shape},
            {'type': 'batch_norm'},
            {'type': 'max_pooling', 'pool_size': (2, 2)},
            {'type': 'dropout', 'rate': 0.25},

            {'type': 'conv2d', 'filters': 128, 'kernel_size': (5, 5), 'activation': 'relu', 'padding': 'same'},
            {'type': 'batch_norm'},
            {'type': 'max_pooling', 'pool_size': (2, 2)},
            {'type': 'dropout', 'rate': 0.25},

            {'type': 'conv2d', 'filters': 512, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
            {'type': 'batch_norm'},
            {'type': 'max_pooling', 'pool_size': (2, 2)},
            {'type': 'dropout', 'rate': 0.25},

            {'type': 'conv2d', 'filters': 512, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'},
            {'type': 'batch_norm'},
            {'type': 'max_pooling', 'pool_size': (2, 2)},
            {'type': 'dropout', 'rate': 0.25},

            {'type': 'flatten'},

            {'type': 'dense', 'units': 256, 'activation': 'relu'},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': 0.25},

            {'type': 'dense', 'units': 512, 'activation': 'relu'},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': 0.25}
        ],
        'output_units': num_classes,
        'output_activation': 'softmax'
    }

    # Set up data generators
    parent_directory = os.path.dirname(os.getcwd())
    train_path = os.path.join(parent_directory, 'data', 'train')
    test_path = os.path.join(parent_directory, 'data', 'test')
    train_generator, validation_generator = setup_data_generators(train_path, test_path)

    # Build, train and evaluate the model
    model = build_train_evaluate(advanced_config, input_shape, num_classes, train_generator, validation_generator, epochs, config_name)
    model.save(os.path.join(parent_directory, 'models','final_model_w_weights.h5'))

if __name__ == "__main__":
    main()
