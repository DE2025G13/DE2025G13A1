from typing import NamedTuple
from kfp import dsl
from kfp.dsl import (
    Input,
    Output,
    Model,
    Dataset,
)

# ==============================================================================
# Component Definitions with Corrected Import Scope
# ==============================================================================

@dsl.component
def data_ingestion(
    bucket_name: str,
    blob_name: str,
    error_log_bucket: str, # Added for remote logging
) -> NamedTuple("outputs", [("raw_dataset", Dataset)]):
    """Downloads data from GCS and outputs it as a KFP Dataset Artifact."""
    # FIX: Import inside the function to fix the NameError
    from kfp.dsl import ContainerSpec
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/data-ingestion:latest',
        command=["python3", "component.py"],
        args=[
            "--bucket-name", bucket_name,
            "--blob-name", blob_name,
            "--output-dataset-path", dsl.OutputPath("raw_dataset"),
            # Pass the new argument for error logging
            "--error-log-bucket", error_log_bucket,
        ],
    )

@dsl.component
def train_test_splitter(
    input_dataset: Input[Dataset]
) -> NamedTuple("outputs", [("training_data", Dataset), ("testing_data", Dataset)]):
    """Splits the input Dataset into training and testing datasets."""
    # FIX: Import inside the function
    from kfp.dsl import ContainerSpec
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/train-test-splitter:latest',
        command=["python3", "component.py"],
        args=[
            "--input-dataset-path",
            dsl.InputPath(input_dataset),
            "--training-data-path",
            dsl.OutputPath("training_data"),
            "--testing-data-path",
            dsl.OutputPath("testing_data"),
        ],
    )

@dsl.component
def train_model(
    image: str,
    training_data: Input[Dataset]
) -> NamedTuple("outputs", [("model", Model)]):
    """A generic training component that accepts a specific container image."""
    # FIX: Import inside the function
    from kfp.dsl import ContainerSpec
    return ContainerSpec(
        image=image,
        command=["python3", "component.py"],
        args=[
            "--training_data_path",
            dsl.InputPath(training_data),
            "--model_artifact_path",
            dsl.OutputPath("model"),
        ],
    )

@dsl.component
def model_evaluator(
    testing_data: Input[Dataset],
    decision_tree_model: Input[Model],
    linear_regression_model: Input[Model],
    logistic_regression_model: Input[Model],
    model_bucket_name: str,
    prod_model_blob: str,
) -> NamedTuple("outputs", [("decision", str), ("best_model_uri", str)]):
    """Evaluates trained models and outputs a deployment decision."""
    # FIX: Import inside the function
    from kfp.dsl import ContainerSpec
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-evaluator:latest',
        command=["python3", "component.py"],
        args=[
            "--testing_data_path",
            dsl.InputPath(testing_data),
            "--decision_tree_model_path",
            dsl.InputPath(decision_tree_model),
            "--linear_regression_model_path",
            dsl.InputPath(linear_regression_model),
            "--logistic_regression_model_path",
            dsl.InputPath(logistic_regression_model),
            "--model_bucket_name",
            model_bucket_name,
            "--prod_model_blob",
            prod_model_blob,
            "--decision",
            dsl.OutputPath("decision"),
            "--best_model_uri",
            dsl.OutputPath("best_model_uri"),
        ],
    )

@dsl.component
def trigger_cd_pipeline(
    project_id: str,
    trigger_id: str,
    new_model_uri: str,
    best_model_name: str,
):
    """Triggers the Cloud Build deployment pipeline."""
    # FIX: Import inside the function
    from kfp.dsl import ContainerSpec
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/trigger-cd:latest',
        command=["python3", "component.py"],
        args=[
            "--project-id", project_id,
            "--trigger-id", trigger_id,
            "--new-model-uri", new_model_uri,
            "--best-model-name", best_model_name,
        ],
    )

# ==============================================================================
# Pipeline Definition
# ==============================================================================

@dsl.pipeline(name='wine-quality-end-to-end-pipeline-v6') # Incremented version
def wine_quality_pipeline(
    project_id: str = "data-engineering-vm",
    data_bucket: str = "yannick-wine-data",
    model_bucket: str = "yannick-wine-models",
    cd_trigger_id: str = "deploy-wine-app-trigger"
):
    # Pass the data bucket name for error logging as well
    ingestion_task = data_ingestion(
        bucket_name=data_bucket,
        blob_name="raw/WineQT.csv",
        error_log_bucket=data_bucket
    )

    split_task = train_test_splitter(
        input_dataset=ingestion_task.outputs["raw_dataset"]
    )

    dt_task = train_model(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-dt:latest',
        training_data=split_task.outputs["training_data"]
    )
    lr_task = train_model(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-lr:latest',
        training_data=split_task.outputs["training_data"]
    )
    logr_task = train_model(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-logr:latest',
        training_data=split_task.outputs["training_data"]
    )

    eval_task = model_evaluator(
        testing_data=split_task.outputs["testing_data"],
        decision_tree_model=dt_task.outputs["model"],
        linear_regression_model=lr_task.outputs["model"],
        logistic_regression_model=logr_task.outputs["model"],
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib",
    )
    eval_task.set_caching_options(enable_caching=False)

    with dsl.If(eval_task.outputs["decision"] == "deploy_new", name="if-new-model-is-better"):
        trigger_cd_pipeline(
            project_id=project_id,
            trigger_id=cd_trigger_id,
            new_model_uri=eval_task.outputs["best_model_uri"],
            best_model_name="best-model"
        )

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("\nPipeline compiled successfully to wine_quality_pipeline.yaml\n")