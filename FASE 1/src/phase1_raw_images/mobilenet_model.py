import tensorflow as tf
from tensorflow.keras import layers, models, applications


def get_data_augmentation_layer():
    """
    Capa de aumento de datos robusta para Fase 1 (Imágenes RGB).
    Importante: NO incluye RandomFlip horizontal para evitar invertir la lateralidad
    y orientación crítica de las señas.
    Augmentation agresiva para cerrar el domain gap con webcam en vivo.
    """
    return tf.keras.Sequential([
        layers.RandomRotation(factor=0.05, fill_mode='nearest'),      # ± ~18 grados (más variación de pose)
        layers.RandomTranslation(height_factor=0.12, width_factor=0.12, fill_mode='nearest'),  # desplazamientos mayores
        layers.RandomZoom(height_factor=(-0.15, 0.15), fill_mode='nearest'),  # zoom más agresivo
        layers.RandomBrightness(factor=0.25),  # mayor variación de iluminación (simula webcam)
        layers.RandomContrast(factor=0.25),    # mayor variación de contraste
    ], name="phase1_data_augmentation")


def build_mobilenetv3_lsc70(input_shape=(224, 224, 3), num_classes=37, trainable_base=False, fine_tune_layers=0):
    """
    Construye el modelo MobileNetV3Large con:
    1. Pipeline de Data Augmentation integrado.
    2. Backbone MobileNetV3Large preentrenado con soporte de congelamiento o fine-tuning selectivo.
    3. Cabeza de clasificación regularizada con Dropout y regularización L2.
    4. BatchNorm SIEMPRE en modo inferencia para preservar estadísticas de ImageNet.
    """
    base_model = applications.MobileNetV3Large(
        input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg'
    )

    if trainable_base:
        if fine_tune_layers > 0:
            # Descongelar únicamente las últimas 'fine_tune_layers' capas
            base_model.trainable = True
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
        else:
            base_model.trainable = True
        # FIX CRÍTICO: Congelar TODAS las capas BatchNormalization del backbone
        # para que no sobreescriban las estadísticas de ImageNet con el dataset pequeño LSC70.
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # 1. Data Augmentation (solo se activa durante training=True)
    aug = get_data_augmentation_layer()
    x = aug(inputs)

    # 2. Preprocesamiento nativo de MobileNetV3
    x = applications.mobilenet_v3.preprocess_input(x)

    # 3. Paso por el backbone convolucional
    # FIX: SIEMPRE training=False para que BatchNorm use estadísticas de ImageNet
    x = base_model(x, training=False)

    # 4. Cabeza densa ampliada con dos capas para mejor discriminación entre 37 clases
    x = layers.Dense(
        512,
        activation='hard_swish',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_features_1"
    )(x)
    x = layers.BatchNormalization(name="bn_features_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)

    x = layers.Dense(
        256,
        activation='hard_swish',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_features_2"
    )(x)
    x = layers.BatchNormalization(name="bn_features_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)

    model = models.Model(inputs, outputs, name="MobileNetV3_LSC70_Phase1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model
