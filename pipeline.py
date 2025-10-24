from typing import NamedTuple
from kfp import dsl
# FIX: Import all necessary types from kfp.dsl
from kfp.dsl import Input, Output, Model, Dataset, ContainerSpec

# ==============================================================================
# Component Definitions
# ==============================================================================

@dsl.component
def data_ingestion_op(
    bucket_name: str,
    blob_name: str,
    error_log_bucket: str
) -> NamedTuple("outputs", [("raw_dataset", Dataset)]):
    """Factory for the data ingestion container component."""
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/data-ingestion:latest',
        command=["python3", "component.py"],
        args=[
            "--bucket-name", bucket_name,
            "--blob-name", blob_name,
            "--output-dataset-path", dsl.OutputPath("raw_dataset"),
            "--error-log-bucket", error_log_bucket,
        ]
    )

@dsl.component
def train_test_splitter_op(
    input_dataset: Input[Dataset]
) -> NamedTuple("outputs", [("training_data", Dataset), ("testing_data", Dataset)]):
    """Factory for the train-test splitter component."""
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/train-test-splitter:latest',
        command=["python3", "component.py"],
        args=[
            "--input-dataset-path", dsl.InputPath(input_dataset),
            "--training-data-path", dsl.OutputPath("training_data"),
            "--testing-data-path", dsl.OutputPath("testing_data"),
        ]
    )

@dsl.component
def train_model_op(
    image: str,
    training_data: Input[Dataset]
) -> NamedTuple("outputs", [("model", Model)]):
    """Factory for a generic model training component."""
    return ContainerSpec(
        image=image,
        command=["python3", "component.py"],
        args=[
            "--training_data_path", dsl.InputPath(training_data),
            "--model_artifact_path", dsl.OutputPath("model"),
        ]
    )

@dsl.component
def model_evaluator_op(
    testing_data: Input[Dataset],
    decision_tree_model: Input[Model],
    linear_regression_model: Input[Model],
    logistic_regression_model: Input[Model],
    model_bucket_name: str,
    prod_model_blob: str,
) -> NamedTuple("outputs", [("decision", str), ("best_model_uri", str), ("best_model_name", str)]): # IMPROVEMENT: Added best_model_name output
    """Factory for the model evaluator component."""
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-evaluator:latest',
        command=["python3", "component.py"],
        args=[
            "--testing_data_path", dsl.InputPath(testing_data),
            "--decision_tree_model_path", dsl.InputPath(decision_tree_model),
            "--linear_regression_model_path", dsl.InputPath(linear_regression_model),
            "--logistic_regression_model_path", dsl.InputPath(logistic_regression_model),
            "--model_bucket_name", model_bucket_name,
            "--prod_model_blob", prod_model_blob,
            "--decision", dsl.OutputPath("decision"),
            "--best_model_uri", dsl.OutputPath("best_model_uri"),
            "--best_model_name", dsl.OutputPath("best_model_name"), # IMPROVEMENT: Added best_model_name output path
        ]
    )

@dsl.component
def trigger_cd_pipeline_op(
    project_id: str,
    trigger_id: str,
    new_model_uri: str,
    best_model_name: str,
):
    """Factory for the CD trigger component."""
    return ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/trigger-cd:latest',
        command=["python3", "component.py"],
        args=[
            "--project-id", project_id,
            "--trigger-id", trigger_id,
            "--new-model-uri", new_model_uri,
            "--best-model-name", best_model_name,
        ]
    )

# ==============================================================================
# Pipeline Definition
# ==============================================================================

@dsl.pipeline(name='wine-quality-end-to-end-pipeline-v9')
def wine_quality_pipeline(
    project_id: str = "data-engineering-vm",
    data_bucket: str = "yannick-wine-data",
    model_bucket: str = "yannick-wine-models",
    cd_trigger_id: str = "deploy-wine-app-trigger"
):
    # Step 1: Ingest data
    ingestion_task = data_ingestion_op(
        bucket_name=data_bucket,
        blob_name="raw/WineQT.csv",
        error_log_bucket=data_bucket
    )

    # Step 2: Split data
    split_task = train_test_splitter_op(
        input_dataset=ingestion_task.outputs["raw_dataset"]
    )

    # Step 3: Train models in parallel
    dt_task = train_model_op(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-dt:latest',
        training_data=split_task.outputs["training_data"]
    ).set_display_name('train-decision-tree')

    lr_task = train_model_op(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-lr:latest',
        training_data=split_task.outputs["training_data"]
    ).set_display_name('train-linear-regression')

    logr_task = train_model_op(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-logr:latest',
        training_data=split_task.outputs["training_data"]
    ).set_display_name('train-logistic-regression')

    # Step 4: Evaluate models
    eval_task = model_evaluator_op(
        testing_data=split_task.outputs["testing_data"],
        decision_tree_model=dt_task.outputs["model"],
        linear_regression_model=lr_task.outputs["model"],
        logistic_regression_model=logr_task.outputs["model"],
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib",
    ).set_caching_options(enable_caching=False)

    # Step 5: Conditional deployment trigger
    with dsl.If(eval_task.outputs["decision"] == "deploy_new", name="if-new-model-is-better"):
        trigger_cd_pipeline_op(
            project_id=project_id,
            trigger_id=cd_trigger_id,
            new_model_uri=eval_task.outputs["best_model_uri"],
            best_model_name=eval_task.outputs["best_model_name"] # IMPROVEMENT: Use dynamic model name
        )

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("\nPipeline compiled successfully to wine_quality_pipeline.yaml\n")