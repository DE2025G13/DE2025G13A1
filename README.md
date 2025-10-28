# MLOps: End-to-End Wine Quality Prediction System

This project implements a fully automated, end-to-end MLOps system on Google Cloud Platform to predict wine quality. It features a complete CI/CD/CT (Continuous Integration/Deployment/Training) workflow that automatically builds, tests, trains, evaluates, and deploys a machine learning model and its serving application.

## Key Features

-   **Automated CI/CD/CT**: The entire lifecycle from code/data commit to production deployment is automated.
-   **Three-Branch Git Strategy**: Code, data, and frontend changes are managed in separate branches (`main`, `dataset`, `frontend`) to trigger specific, efficient workflows without unnecessary rebuilds.
-   **Serverless Architecture**: Built entirely on serverless GCP services like Cloud Build, Vertex AI, and Cloud Run for scalability and cost-efficiency.
-   **Automated Quality Gates**:
    -   **Data**: New datasets are automatically validated against 17 quality checks before being used for training.
    -   **Model**: A new model is only promoted to production if it outperforms the current one on both accuracy and F1-score metrics.
-   **Parallel Model Training**: The system trains Random Forest, XGBoost, and SVM models in parallel, selecting the best candidate via cross-validation.

## Architecture and Automation

The system's automation is orchestrated by Cloud Build triggers linked to three specific Git branches.

1.  **`dataset` Branch (Continuous Training)**
    -   **Trigger**: A push of a new `wine.csv` to this branch.
    -   **Action**:
        1.  **Validate Data**: Cloud Build runs 17 data quality tests.
        2.  **Store Commit Hash**: If validation passes, the new dataset's Git commit hash is saved to Cloud Storage.
        3.  **Trigger Pipeline**: A Vertex AI training pipeline is automatically started, using the newly validated data.

2.  **`main` Branch (Continuous Integration)**
    -   **Trigger**: A push with changes to ML pipeline components.
    -   **Action**:
        1.  **Run Unit Tests**: All 7 pipeline components are tested.
        2.  **Build Docker Images**: New Docker images for the components are built and pushed to Artifact Registry.
        3.  **Run Pipeline**: The full Vertex AI pipeline is executed using the latest validated dataset to ensure integrity.

3.  **`frontend` Branch (Continuous Deployment)**
    -   **Trigger**: A push with changes to the API or UI code.
    -   **Action**:
        1.  **Run Integration Tests**: A suite of 24 tests validates the API and UI.
        2.  **Build Images**: Docker images for the `prediction-api` and `prediction-ui` are built.
        3.  **Deploy to Cloud Run**: Both services are automatically deployed to Cloud Run.

## Technology Stack

-   **Orchestration**: Vertex AI Pipelines, Cloud Build
-   **Containerization**: Docker, Google Artifact Registry
-   **Serving**: Cloud Run (for a private API and a public UI)
-   **Storage**: Google Cloud Storage (for models, artifacts, and configs)
-   **ML Frameworks**: Scikit-learn, XGBoost
-   **Version Control**: GitHub

## Vertex AI ML Pipeline Flow

When triggered, the ML pipeline executes the following serverless steps:

1.  **Data Ingestion**: Downloads the validated `wine.csv` from GitHub using the stored commit hash.
2.  **Train-Test Split**: Splits the data into stratified training and testing sets.
3.  **Train Models**: Trains Random Forest, XGBoost, and SVM models in parallel.
4.  **Evaluate & Decide**:
    -   Selects the best-performing candidate model using 5-fold cross-validation.
    -   Compares this candidate against the current production model on a hold-out test set.
5.  **Trigger Deployment**: If the new model is superior, it is uploaded to Cloud Storage as the new production model, and the `frontend` branch deployment trigger is called to update the live application.