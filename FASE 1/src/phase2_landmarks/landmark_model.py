import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_landmark_classifier(input_shape=(224, 224, 3), num_classes=70, trainable_base=False):
    """
    Construye el clasificador MobileNetV3Large para la Fase 2 (Imágenes de Landmarks / Esqueleto 224x224x3).
    Garantiza paridad arquitectónica total con la Fase 1 (mismo backbone, misma cabeza densa).
    """
    base_model = applications.MobileNetV3Large(
        input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg'
    )
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=input_shape)
    x = applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=trainable_base)

    x = layers.Dense(512, activation='hard_swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Alias explícito
build_landmark_mobilenetv3 = build_landmark_classifier

