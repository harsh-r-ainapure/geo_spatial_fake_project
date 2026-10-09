"""
Phase 2: Bi-Temporal Siamese CNN for Structural Damage Classification
=======================================================================
Architecture:
- Shared 4-stage Conv2D CNN Encoder (256-d bottleneck per branch)
- Triplet Feature Fusion: [F_before, F_after, |F_before - F_after|] (768-d)
- Dense Classification Head -> 3-Class Softmax:
    0: No Damage
    1: Damaged (Minor + Major)
    2: Destroyed
"""

import tensorflow as tf

keras = tf.keras
layers = keras.layers
Model = keras.Model


# @keras.saving.register_keras_serializable(package="siamese_cnn")
class AbsDifference(layers.Layer):
    """
    Serializable absolute element-wise difference layer.
    Replaces Lambda to allow safe model saving and loading.
    Computes: |F_before - F_after|
    """
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def get_config(self):
        return super().get_config()


def build_shared_encoder(input_shape=(256, 256, 3), feature_dim=256, name="shared_cnn_encoder") -> Model:
    """
    Builds the single convolutional backbone whose weights are shared
    between the Before (T1) and After (T2) branches.
    """
    inputs = layers.Input(shape=input_shape, name="encoder_input")

    # Block 1: 256x256x3 -> 128x128x32
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2: 128x128x32 -> 64x64x64
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3: 64x64x64 -> 32x32x128
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Block 4: 32x32x128 -> 16x16x256
    x = layers.Conv2D(feature_dim, kernel_size=(3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Global Average Pooling: 16x16x256 -> 256-dimensional feature vector
    features = layers.GlobalAveragePooling2D(name="gap")(x)

    encoder = Model(inputs=inputs, outputs=features, name=name)
    return encoder


def build_siamese_damage_classifier(
    input_shape=(256, 256, 3),
    feature_dim=256,
    num_classes=3,
    dropout_rate=0.3
) -> Model:
    """
    Constructs the complete Siamese CNN model:
    - Feeds Before and After images through the SAME shared encoder.
    - Fuses features via: [F_before, F_after, |F_before - F_after|].
    - Classifies into 3 structural damage classes via Softmax.
    """
    # 1. Bi-temporal Inputs
    input_before = layers.Input(shape=input_shape, name="before_image")
    input_after = layers.Input(shape=input_shape, name="after_image")

    # 2. Shared Encoder (Crucial: Instantiated ONCE so weights are identical)
    encoder = build_shared_encoder(input_shape=input_shape, feature_dim=feature_dim)

    # Both images pass through the identical feature extractor
    f_before = encoder(input_before)   # Shape: (batch_size, 256)
    f_after = encoder(input_after)     # Shape: (batch_size, 256)

    # 3. Explicit Learned Difference using registered serializable layer
    diff = AbsDifference(name="abs_difference")([f_before, f_after])  # Shape: (batch_size, 256)

    # 4. Triplet Feature Fusion: [What was there, What is there now, What changed]
    fusion = layers.Concatenate(name="feature_fusion")([f_before, f_after, diff])
    # Total dimension: 256 + 256 + 256 = 768

    # 5. Classification Head
    x = layers.Dense(256, name="dense_1")(fusion)
    x = layers.BatchNormalization(name="bn_dense_1")(x)
    x = layers.ReLU(name="relu_dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(128, name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_dense_2")(x)
    x = layers.ReLU(name="relu_dense_2")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="dropout_2")(x)

    # 6. Final 3-class prediction: [No Damage (0), Damaged (1), Destroyed (2)]
    outputs = layers.Dense(num_classes, activation="softmax", name="damage_probabilities")(x)

    # Build end-to-end model
    model = Model(
        inputs=[input_before, input_after],
        outputs=outputs,
        name="SiameseDamageClassifier"
    )

    return model


if __name__ == "__main__":
    # Instantiate and inspect model structure
    model = build_siamese_damage_classifier()
    model.summary()

    # Verify forward pass with synthetic batch of size 2
    dummy_before = tf.random.uniform((2, 256, 256, 3), minval=0.0, maxval=1.0)
    dummy_after = tf.random.uniform((2, 256, 256, 3), minval=0.0, maxval=1.0)

    predictions = model([dummy_before, dummy_after])
    print("\nForward pass test successful!")
    print(f"Output batch shape: {predictions.shape} (Expected: (2, 3))")
    print(f"Sample prediction probabilities: {predictions.numpy()[0]}")
    print(f"Sum of probabilities: {predictions.numpy()[0].sum():.4f}")