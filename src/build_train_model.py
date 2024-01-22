# model_training.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def setup_data_generators(train_path, test_path, img_size=48, batch_size=64):
    """
    Sets up training and validation data generators with specified parameters.
    """
    datagen_params = {
        'target_size': (img_size, img_size),
        'color_mode': 'grayscale',
        'batch_size': batch_size,
        'class_mode': 'categorical'
    }

    datagen_train = ImageDataGenerator(horizontal_flip=True)
    train_generator = datagen_train.flow_from_directory(train_path, shuffle=True, **datagen_params)

    datagen_validation = ImageDataGenerator(horizontal_flip=True)
    validation_generator = datagen_validation.flow_from_directory(test_path, shuffle=False, **datagen_params)

    return train_generator, validation_generator

def build_advanced_model(config, input_shape):
    """
    Builds a CNN model based on the specified configuration.
    """
    model = Sequential()

    for layer in config['layers']:
        if layer['type'] == 'conv2d':
            if 'input_shape' in layer:
                model.add(Conv2D(layer['filters'], layer['kernel_size'], activation=layer['activation'], padding=layer['padding'], input_shape=layer['input_shape']))
            else:
                model.add(Conv2D(layer['filters'], layer['kernel_size'], activation=layer['activation'], padding=layer['padding']))

        elif layer['type'] == 'batch_norm':
            model.add(BatchNormalization())

        elif layer['type'] == 'max_pooling':
            model.add(MaxPooling2D(pool_size=layer['pool_size']))

        elif layer['type'] == 'dropout':
            model.add(Dropout(layer['rate']))

        elif layer['type'] == 'flatten':
            model.add(Flatten())

        elif layer['type'] == 'dense':
            model.add(Dense(layer['units'], activation=layer['activation']))

    model.add(Dense(config['output_units'], activation=config['output_activation']))

    return model

def train_model(model, train_generator, validation_generator, epochs, model_name):
    """
    Train the CNN model and save the weights with a custom name.
    """
    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size

    parent_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(parent_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"{model_name}_model_weights.h5"
    model_path = os.path.join(model_dir, model_filename)

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto')
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1)
    callbacks = [early_stopping, checkpoint, reduce_lr]

    history = model.fit(
        x=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    return history
