from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model, Metrics, OutputPath

# This is the base path in our Artifact Registry where the component images are stored.
IMAGE_REGISTRY_PATH = "europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo"

@dsl.container_component
def data_ingestion_op(input_data_gcs_path: str, raw_dataset: Output[Dataset]):
    # Defines the data ingestion step, which reads from GCS
    return dsl.ContainerSpec(
        image=f"{IMAGE_REGISTRY_PATH}/data-ingestion:latest",
        command=["python3", "component.py"],
        args=[
            "--input-data-gcs-path", input_data_gcs_path,
            "--output-dataset-path", raw_dataset.path
        ]
    )

@dsl.container_component
def train_test_splitter_op(input_dataset: Input[Dataset], training_data: Output[Dataset], testing_data: Output[Dataset]):
    # Defines the component that splits our data into training and testing sets.
    return dsl.ContainerSpec(
        image=f"{IMAGE_REGISTRY_PATH}/train-test-splitter:latest",
        command=["python3", "component.py"],
        args=["--input-dataset-path", input_dataset.path, "--training-data-path", training_data.path, "--testing-data-path", testing_data.path]
    )

@dsl.container_component
def train_model_op(image_name: str, training_data: Input[Dataset], model_artifact: Output[Model]):
    # This is a generic trainer; we can reuse it for any model just by changing the image name.
    return dsl.ContainerSpec(
        image=f"{IMAGE_REGISTRY_PATH}/{image_name}:latest",
        command=["python3", "component.py"],
        args=["--training_data_path", training_data.path, "--model_artifact_path", model_artifact.path]
    )
    
@dsl.container_component
def model_evaluator_op(
    training_data: Input[Dataset],
    testing_data: Input[Dataset],
    random_forest_model: Input[Model],
    xgboost_model: Input[Model],
    svm_model: Input[Model],
    model_bucket_name: str,
    prod_model_blob: str,
    decision: OutputPath(str),
    best_model_uri: OutputPath(str),
    metrics: Output[Metrics],
):
    # Defines our main evaluation component, which takes all models and data as input.
    return dsl.ContainerSpec(
        image=f"{IMAGE_REGISTRY_PATH}/model-evaluator:latest",
        command=["python3", "component.py"],
        args=[
            "--training_data_path", training_data.path,
            "--testing_data_path", testing_data.path,
            "--random_forest_model_path", random_forest_model.path,
            "--xgboost_model_path", xgboost_model.path,
            "--svm_model_path", svm_model.path,
            "--model_bucket_name", model_bucket_name,
            "--prod_model_blob", prod_model_blob,
            "--decision_path", decision,
            "--best_model_uri_path", best_model_uri,
            "--metrics_path", metrics.path,
        ]
    )

@dsl.container_component
def trigger_cd_pipeline_op(project_id: str, trigger_id: str, new_model_uri: str, region: str = "europe-west4"):
    # This component's only job is to start our CD pipeline if a new model is chosen.
    return dsl.ContainerSpec(
        image=f"{IMAGE_REGISTRY_PATH}/trigger-cd:latest",
        command=["python3", "component.py"],
        args=[
            "--project-id", project_id, 
            "--trigger-id", trigger_id, 
            "--new-model-uri", new_model_uri,
            "--region", region
        ]
    )

@dsl.pipeline(name="wine-quality-git-triggered-pipeline")
def wine_quality_pipeline(
    # GCS path to the dataset (uploaded by Cloud Build)
    input_data_gcs_path: str = "gs://yannick-pipeline-root/datasets/wine-latest.csv",
    project_id: str = "data-engineering-vm",
    model_bucket: str = "yannick-wine-models",
    cd_trigger_id: str = "b4b2dba0-3797-495c-a78f-eaea06982348",
    region: str = "europe-west4"
):
    # This function defines the graph of our ML pipeline, connecting all the steps.
    ingestion_task = data_ingestion_op(input_data_gcs_path=input_data_gcs_path)
    split_task = train_test_splitter_op(input_dataset=ingestion_task.outputs["raw_dataset"])
    # These three training tasks will all run in parallel to save time.
    rf_task = train_model_op(image_name="model-trainer-rf", training_data=split_task.outputs["training_data"]).set_display_name("Train-Random-Forest")
    xgb_task = train_model_op(image_name="model-trainer-xgb", training_data=split_task.outputs["training_data"]).set_display_name("Train-XGBoost")
    svm_task = train_model_op(image_name="model-trainer-svm", training_data=split_task.outputs["training_data"]).set_display_name("Train-SVM")
    # The evaluation step waits for all training jobs to finish before it starts.
    eval_task = model_evaluator_op(
        training_data=split_task.outputs["training_data"],
        testing_data=split_task.outputs["testing_data"],
        random_forest_model=rf_task.outputs["model_artifact"],
        xgboost_model=xgb_task.outputs["model_artifact"],
        svm_model=svm_task.outputs["model_artifact"],
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib",
    ).set_caching_options(enable_caching=False)
    # This 'If' block creates a conditional branch inside of our pipeline.
    with dsl.If(eval_task.outputs["decision"] == "deploy_new", name="if-new-model-is-better"):
        trigger_cd_pipeline_op(
            project_id=project_id,
            trigger_id=cd_trigger_id,
            new_model_uri=eval_task.outputs["best_model_uri"],
            region=region
        )

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(wine_quality_pipeline, "wine_quality_pipeline_git_triggered.yaml")
    print("Pipeline has been compiled successfully to wine_quality_pipeline_git_triggered.yaml.")