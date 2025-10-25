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

@dsl.pipeline(name='wine-quality-pipeline')
def wine_quality_pipeline(
    data_bucket: str = "yannick-wine-data"
):
    """A pipeline that ingests, splits, and trains three models in parallel."""
    
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

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("Pipeline compiled successfully to wine_quality_pipeline.yaml")