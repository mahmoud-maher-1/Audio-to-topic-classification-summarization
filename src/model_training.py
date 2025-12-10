import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers


def build_distributed_model(input_shape, num_classes):
    """
    Builds the 20-class classification model.
    Optimizer: Adam (lr=1e-3)
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,), dtype=tf.float32),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_distributed_model(model, X_train, y_train):
    """
    Trains the distributed model with notebook-specific settings:
    - ReduceLROnPlateau: factor=0.35, patience=3
    - EarlyStopping: monitor='val_accuracy', patience=8
    - Batch size: 32
    - Validation split: 0.2
    - Epochs: 100
    """
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.35, patience=3, min_lr=1e-8, verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    return history


def build_generalized_model(input_shape, num_classes):
    """
    Builds the 6-class classification model.
    Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,), dtype=tf.float32),
        layers.Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='gelu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_generalized_model(model, X_train, y_train):
    """
    Trains the generalized model with notebook-specific settings:
    - ReduceLROnPlateau: factor=0.5, patience=2
    - EarlyStopping: monitor='val_loss', patience=4
    - Batch size: 256
    - Validation split: 0.1
    - Epochs: 50
    """
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    return history