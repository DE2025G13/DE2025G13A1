# MLOps Project: End-to-End Wine Quality Prediction

Hey there! Welcome to our MLOps project. The goal here is to build a full, automated system that can train a machine learning model to predict the quality of wine and then deploy it as a web application.

This isn't just about building a model; it's about building the entire automated factory around it.

## What's This Project About?

The core idea is to predict a wine's quality score based on its chemical properties and whether it's red or white. We've set up a complete MLOps pipeline using Google Cloud that handles everything from data changes to model deployment automatically.

### The Dataset (`wine.csv`)

The data we're using is a combined dataset of red and white variants of the Portuguese "Vinho Verde" wine. This makes the prediction task more interesting!

- **Filename:** `wine.csv`
- **Location:** `dataset/wine.csv`
- **What's inside:** It has about 6,500 rows.
- **Features:** It includes a new `type` column ("red" or "white") and 11 chemical measurements like `fixed_acidity`, `volatile_acidity`, `alcohol`, etc.
- **Target:** The `quality` column is what we're trying to predict.
- **Versioning:** We keep the dataset directly in a separate `dataset` Git branch. This is super handy because every time we update the data and push to that branch, it automatically triggers a new training pipeline run. It also gives us a full version history of the datasets we've used for training.

## Our MLOps Architecture on Google Cloud

We're using a bunch of different tools on Google Cloud Platform (GCP) to make this all work together. Here’s a quick rundown of our setup.

### 1. Google Cloud Storage (GCS) Buckets

We use GCS to store all the "stuff" (artifacts) our pipeline produces. We've set up a few different buckets to keep things organized:

- **`yannick-wine-models`**: This is where we store our trained models. The final, best model that's currently in production lives here.
- **`yannick-pipeline-root`**: This is the main "workspace" for our Vertex AI pipeline. Every time the pipeline runs, it creates a new folder here to store all the intermediate data, logs, and artifacts for that specific run.

### 2. Artifact Registry

This is our private Docker Hub for the project.

- **Repository Name:** `yannick-wine-repo`
- **What it does:** Every component of our ML pipeline (like data-ingestion, model-trainers, and the evaluator) and our final applications (`prediction-api`, `prediction-ui`) are packaged into Docker images. This registry stores all those images.

### 3. IAM Permissions (Who Can Do What)

Getting permissions right is super important. We made sure the "robot" accounts (Service Accounts) used by Google Cloud had the right roles:

- **Compute Engine default service account:** This is the account Vertex AI uses to run our pipeline steps. It has roles like `Vertex AI User` and `Storage Object Admin`.
- **Cloud Build service account:** This is the account Cloud Build uses. It has roles like `Cloud Run Admin` (so it can deploy our apps) and `Artifact Registry Writer` (so it can push Docker images).

### 4. Cloud Build Triggers (The Automation Magic)

The triggers are the heart of our automation. We have two main triggers that watch our GitHub repository:

#### Trigger 1: `build-and-run-on-code-change` (The CI Trigger)
- **Watches:** Any push to the `main` branch.
- **Action:** When we push new code, this trigger runs `components_cloudbuild.yaml`. This file first rebuilds all our Docker container images and then automatically starts a Vertex AI pipeline run to validate that our new code works correctly with the current dataset.

#### Trigger 2: `trigger-training-on-data-change` (The CT Trigger)
- **Watches:** A push to the `dataset` branch, but *only* if files inside the `dataset/` folder have changed.
- **Action:** When we push a new `wine.csv`, this trigger runs `run_training_pipeline.yaml`. Its only job is to start a new Vertex AI training pipeline run using the new data.

### 5. The Vertex AI Pipeline

This is where the actual machine learning happens. It's a graph of steps that runs serverlessly on Vertex AI.

1.  **Ingest Data:** Grabs the `wine.csv` from the Git repo.
2.  **Split Data:** Preprocesses the data (encodes the 'type' column) and splits it into training and testing sets.
3.  **Train Models (in parallel):** Trains three different models at the same time: `Random Forest`, `XGBoost`, and `SVM`.
4.  **Evaluate & Decide:**
    - It uses **k-fold cross-validation** to robustly pick the best-performing model out of the three candidates.
    - It then evaluates this "champion" model on the hold-out test set and compares its score to the model currently in production.
    - If the new one is better, it decides to deploy.
5.  **Trigger CD (Conditional):** If the decision is to deploy, this final step kicks off our deployment pipeline.

### 6. The Deployment Pipeline (CD)

This pipeline, defined in `deployment_cloudbuild.yaml`, gets our model and apps live.

1.  **Copy Model:** It copies the winning model to the final production location in GCS.
2.  **Build & Deploy API:** It builds and deploys the `prediction-api` to Cloud Run as a private service.
3.  **Build & Deploy UI:** It builds and deploys the `prediction-ui` to Cloud Run as a public web application, telling it the private URL of the new API.
