import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, dropout_rate=0.5):
        # Freezing: the original code set `model.trainable = False` inside the loop,
        # which froze the WHOLE model even for partial freezing. Set it per layer instead.
        if freeze_all:
            model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # Keep the last `freeze_till` layers trainable, freeze everything before them
            model.trainable = True
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
        else:
            model.trainable = True

        x = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dropout(dropout_rate)(x)  # reduces overfitting
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        # freeze_till=4 leaves block5 (block5_conv1/2/3 + block5_pool) trainable
        # for fine-tuning; all earlier VGG16 layers stay frozen.
        # Use freeze_all=True, freeze_till=None to go back to a fully frozen base.
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=4,
            learning_rate=self.config.params_learning_rate,
            dropout_rate=0.5
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)