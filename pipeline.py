from kfp import dsl
from kfp.dsl import Dataset, Input, Output

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

@dsl.pipeline(name='wine-quality-pipeline')
def wine_quality_pipeline(
    data_bucket: str = "yannick-wine-data"
):
    """A two-step pipeline to ingest and split data."""
    
    ingestion_task = data_ingestion_op(
        bucket_name=data_bucket,
        blob_name="raw/WineQT.csv"
    )

    split_task = train_test_splitter_op(
        input_dataset=ingestion_task.outputs["raw_dataset"]
    )
    # ----------------------------------------

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("Pipeline compiled successfully to wine_quality_pipeline.yaml")