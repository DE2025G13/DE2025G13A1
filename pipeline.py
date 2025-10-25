from kfp import dsl
from kfp.dsl import Dataset, Output

# Replace 'data-engineering-vm' with your GCP Project ID
IMAGE_REGISTRY_PATH = "europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo"

@dsl.container_component
def data_ingestion_op(
    bucket_name: str,
    blob_name: str,
    raw_dataset: Output[Dataset],
):
    """
    This is the factory component. 
    It defines the interface for the containerized component.
    """
    return dsl.ContainerSpec(
        image=f'{IMAGE_REGISTRY_PATH}/data-ingestion:latest',
        command=[
            "python3", "component.py"
        ],
        args=[
            "--bucket-name", bucket_name,
            "--blob-name", blob_name,
            "--output-dataset-path", raw_dataset.path,
        ]
    )

@dsl.pipeline(name='minimal-ingestion-pipeline')
def minimal_pipeline(
    data_bucket: str = "yannick-wine-data"
):
    """A minimal one-step pipeline to test data ingestion."""
    
    # Run the data ingestion component
    ingestion_task = data_ingestion_op(
        bucket_name=data_bucket,
        blob_name="raw/WineQT.csv"
    )

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=minimal_pipeline,
        package_path='minimal_pipeline.yaml'
    )
    print("Minimal pipeline compiled successfully to minimal_pipeline.yaml")