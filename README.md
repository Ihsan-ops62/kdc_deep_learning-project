# Kidney Disease Classification from CT Images

This project trains a convolutional neural network to classify kidney CT images into four categories: **Cyst**, **Normal**, **Stone**, and **Tumor**. It uses transfer learning with VGG16 and provides a Flask web app for uploading an image and viewing the predicted class and model scores.

> This project is for educational and research use. Predictions are not a medical diagnosis and should not be used to make care decisions.

## Features

- Four-class image classification: Cyst, Normal, Stone, and Tumor.
- VGG16 transfer learning with configurable image size, batch size, epochs, augmentation, and learning rate.
- DVC pipeline for data ingestion, base-model preparation, training, and evaluation.
- Flask interface with image preview, drag-and-drop upload, predicted class, and scores for all classes.
- Optional GitHub Actions workflow for building and publishing a Docker image to Amazon ECR and running deployment commands on a self-hosted runner.

## Dataset

Place the extracted dataset in `artifacts/data_ingestion/kidney_ct_scan_image` with one subfolder per class:

```text
artifacts/data_ingestion/kidney_ct_scan_image/
├── Cyst/
├── Normal/
├── Stone/
└── Tumor/
```

The folders should contain the corresponding image files. Training uses Keras `flow_from_directory`, which assigns class indices alphabetically. The prediction pipeline uses the same order: `Cyst`, `Normal`, `Stone`, `Tumor`.

## Setup

Create and activate your environment, then install the project dependencies:

```powershell
conda create -n kdkhan python=3.10 -y
conda activate kdkhan
pip install -r requirements.txt
```

The training code logs to MLflow. Configure `MLFLOW_TRACKING_URI` in a local `.env` file if your setup requires a tracking server.

## Train and evaluate

Run the DVC pipeline from the repository root:

```powershell
dvc repro
```

The trained model is saved at `artifacts/training/model.h5`; evaluation metrics are written to `scores.json`. You can also run the Python pipeline directly with `python main.py`.

## Run the web app

From the repository root, with the environment activated and the trained model available:

```powershell
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080), choose a CT image, and select **Analyze image**. Set `MODEL_PATH` if the model is stored somewhere other than `artifacts/training/model.h5`.

### API routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serves the image-classification interface. |
| `GET` | `/health` | Returns a basic application health status. |
| `POST` | `/predict` | Accepts an image upload in the `image` multipart field and returns the predicted class and class scores. It also accepts JSON containing a base64 `image` value. |
| `POST` | `/train` | Runs `main.py` to execute the training and evaluation pipeline. |

## Configuration

- `config/config.yaml` contains dataset and artifact paths.
- `params.yaml` contains model and training parameters.
- The prediction pipeline expects the model outputs to follow the alphabetical class order used during training.

## DVC remote storage

`dvc repro` can run locally without a DVC remote. To upload DVC-managed artifacts with `dvc push`, configure a remote in your DVC project first. Keep storage credentials in local DVC configuration or a secrets manager rather than committing them to Git.

## GitHub Actions and AWS

The workflow in `.github/workflows/main.yml` builds a Docker image, pushes it to Amazon ECR, and runs deployment commands on a self-hosted GitHub Actions runner. Configure the repository secrets used by the workflow, including `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY_NAME`, and `AWS_ECR_LOGIN_URI`. The self-hosted runner needs Docker and access to the target AWS resources.


Ihsan Ullah