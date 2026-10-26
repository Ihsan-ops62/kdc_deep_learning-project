import os
import tensorflow as tf
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from cnnClassifier.entity.config_entity import TrainingConfig

load_dotenv()


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False,
        )
        # Adam converges much faster than plain SGD for VGG16 transfer learning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
            run_eagerly=False,
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            # Gentler augmentation, better suited to CT scans
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is missing; configure it in the project .env file.")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("kidney-disease-classification")

        with mlflow.start_run(run_name="kidney-cnn-training"):
            mlflow.log_params({
                "epochs": self.config.params_epochs,
                "batch_size": self.config.params_batch_size,
                "learning_rate": self.config.params_learning_rate,
                "optimizer": "Adam",
                "image_size": str(self.config.params_image_size),
                "augmentation": self.config.params_is_augmentation,
                "classes": self.train_generator.num_classes,
                "train_images": self.train_generator.samples,
                "validation_images": self.valid_generator.samples,
            })

            history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
            )

            for metric_name, values in history.history.items():
                for epoch, value in enumerate(values, start=1):
                    mlflow.log_metric(metric_name, float(value), step=epoch)

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model,
            )