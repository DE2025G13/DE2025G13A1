from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model

IMAGE_REGISTRY_PATH = "europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo"

@dsl.container_component
def data_ingestion_op(
    bucket_name: str,
    blob_name: str,
    raw_dataset: Output[Dataset],
):
    """Factory for the data ingestion container component."""
    return dsl.ContainerSpec(
        image=f'{IMAGE_REGISTRY_PATH}/data-ingestion:latest',
        command=["python3", "component.py"],
        args=[
            "--bucket-name", bucket_name,
            "--blob-name", blob_name,
            "--output-dataset-path", raw_dataset.path,
        ]
    )

@dsl.container_component
def train_test_splitter_op(
    input_dataset: Input[Dataset],
    training_data: Output[Dataset],
    testing_data: Output[Dataset],
):
    """Factory for the train-test splitter component."""
    return dsl.ContainerSpec(
        image=f'{IMAGE_REGISTRY_PATH}/train-test-splitter:latest',
        command=["python3", "component.py"],
        args=[
            "--input-dataset-path", input_dataset.path,
            "--training-data-path", training_data.path,
            "--testing-data-path", testing_data.path,
        ]
    )

@dsl.container_component
def train_model_op(
    image_name: str,
    training_data: Input[Dataset],
    model_artifact: Output[Model],
):
    """Factory for a generic model training component."""
    return dsl.ContainerSpec(
        image=f'{IMAGE_REGISTRY_PATH}/{image_name}:latest',
        command=["python3", "component.py"],
        args=[
            "--training_data_path", training_data.path,
            "--model_artifact_path", model_artifact.path,
        ]
    )
    
@dsl.container_component
def model_evaluator_op(
    testing_data: Input[Dataset],
    decision_tree_model: Input[Model],
    linear_regression_model: Input[Model],
    logistic_regression_model: Input[Model],
    model_bucket_name: str,
    prod_model_blob: str,
    decision: Output[str],
    best_model_uri: Output[str],
):
    """Factory for the model evaluator component."""
    return dsl.ContainerSpec(
        image=f'{IMAGE_REGISTRY_PATH}/model-evaluator:latest',
        command=["python3", "component.py"],
        args=[
            "--testing_data_path", testing_data.path,
            "--decision_tree_model_path", decision_tree_model.path,
            "--linear_regression_model_path", linear_regression_model.path,
            "--logistic_regression_model_path", logistic_regression_model.path,
            "--model_bucket_name", model_bucket_name,
            "--prod_model_blob", prod_model_blob,
            "--decision", decision.path,
            "--best_model_uri", best_model_uri.path,
        ]
    )

@dsl.container_component
def trigger_cd_pipeline_op(
    project_id: str,
    trigger_id: str,
    new_model_uri: str,
):
    """Factory for the CD trigger component."""
    return dsl.ContainerSpec(
        image=f'{IMAGE_REGISTRY_PATH}/trigger-cd:latest',
        command=["python3", "component.py"],
        args=[
            "--project-id", project_id,
            "--trigger-id", trigger_id,
            "--new-model-uri", new_model_uri,
        ]
    )

@dsl.pipeline(name='wine-quality-end-to-end-pipeline')
def wine_quality_pipeline(
    project_id: str = "data-engineering-vm",
    data_bucket: str = "yannick-wine-data",
    model_bucket: str = "yannick-wine-models",
    cd_trigger_id: str = "deploy-wine-app-trigger"
):
    """A full MLOps pipeline for training, evaluating, and deploying a wine quality model."""
    
    ingestion_task = data_ingestion_op(
        bucket_name=data_bucket,
        blob_name="raw/WineQT.csv"
    )

    split_task = train_test_splitter_op(
        input_dataset=ingestion_task.outputs["raw_dataset"]
    )

    dt_task = train_model_op(
        image_name='model-trainer-dt',
        training_data=split_task.outputs["training_data"]
    ).set_display_name('Train-Decision-Tree')

    lr_task = train_model_op(
        image_name='model-trainer-lr',
        training_data=split_task.outputs["training_data"]
    ).set_display_name('Train-Linear-Regression')

    logr_task = train_model_op(
        image_name='model-trainer-logr',
        training_data=split_task.outputs["training_data"]
    ).set_display_name('Train-Logistic-Regression')
    
    eval_task = model_evaluator_op(
        testing_data=split_task.outputs["testing_data"],
        decision_tree_model=dt_task.outputs["model_artifact"],
        linear_regression_model=lr_task.outputs["model_artifact"],
        logistic_regression_model=logr_task.outputs["model_artifact"],
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib",
    ).set_caching_options(enable_caching=False)

    with dsl.If(eval_task.outputs["decision"] == "deploy_new", name="if-new-model-is-better"):
        trigger_cd_pipeline_op(
            project_id=project_id,
            trigger_id=cd_trigger_id,
            new_model_uri=eval_task.outputs["best_model_uri"],
        )

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("\nPipeline compiled successfully to wine_quality_pipeline.yaml\n")