import kfp
from kfp import dsl
from kfp.dsl import Output, Artifact, OutputPath

@dsl.container_component
def canary_test():
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/ml-services/canary-test:latest'
    )

@dsl.container_component
def auto_labeler(
    raw_bucket_name: str,
    processed_bucket_name: str,
    processed_prefix: str,
    labeled_dataset: Output[Artifact]
):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/ml-services/auto-labeler:latest',
        command=[
            "python3", "component.py",
            "--raw-bucket-name", raw_bucket_name,
            "--processed-bucket-name", processed_bucket_name,
            "--processed-prefix", processed_prefix,
        ]
    )

@dsl.container_component
def dataset_splitter(
    bucket_name: str,
    prefix: str,
    data_yaml_artifact: Output[Artifact]
):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/ml-services/dataset-splitter:latest',
        command=[
            "python3", "component.py",
            "--bucket-name", bucket_name,
            "--prefix", prefix,
        ]
    )

@dsl.container_component
def yolo_trainer(
    data_yaml_uri: str,
    output_bucket_name: str,
    output_blob_name: str,
    model_artifact: Output[Artifact],
    image_uri: str = 'europe-west4-docker.pkg.dev/data-engineering-vm/ml-services/yolo-trainer:latest'
):
    return dsl.ContainerSpec(
        image=image_uri,
        command=[
            "python", "component.py",
            "--data-yaml-uri", data_yaml_uri,
            "--output-bucket-name", output_bucket_name,
            "--output-blob_name", output_blob_name
        ]
    )

@dsl.container_component
def model_evaluator(
    data_yaml_uri: str,
    new_model_uri: str,
    prod_model_uri: str,
    decision: OutputPath(str)
):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/ml-services/model-evaluator:latest',
        command=[
            "bash", "-c",
            f"python component.py --data-yaml-uri {data_yaml_uri} --new-model-uri {new_model_uri} --prod-model-uri {prod_model_uri} > {decision}"
        ]
    )

@dsl.container_component
def trigger_cd_pipeline(
    project_id: str,
    trigger_id: str,
    new_model_uri: str
):
    cmd = (
        'set -e && '
        'ACCESS_TOKEN=$(gcloud auth print-access-token) && '
        'curl -X POST -H "Authorization: Bearer $ACCESS_TOKEN" '
        '-H "Content-Type: application/json" '
        f'"https://cloudbuild.googleapis.com/v1/projects/{project_id}/triggers/{trigger_id}:run" '
        f'-d \'{{"substitutions":{{"_NEW_MODEL_URI":"{new_model_uri}"}}}}\''
    )
    return dsl.ContainerSpec(
        image='google/cloud-sdk:slim',
        command=['bash', '-c', cmd]
    )

@dsl.pipeline(
    name='yolo-glasses-detection-pipeline-CANARY-TEST',
    description='A minimal test to see if a simple container can run on a GPU worker.'
)
def object_detection_pipeline(
    project_id: str = "data-engineering-vm",
    pipeline_root: str = "gs://glasses-temp/pipeline-root"
):
    # Step 1: Run only the canary test task
    canary_task = canary_test()

    # Step 2: Assign a GPU to the task to replicate the failure conditions
    canary_task.set_gpu_limit(1)
    canary_task.set_accelerator_type('NVIDIA_TESLA_T4')


if __name__ == '__main__':
    from kfp.compiler import Compiler
    Compiler().compile(
        pipeline_func=object_detection_pipeline,
        package_path='object_detection_pipeline.yaml'
    )
    print("Canary Test Pipeline compiled successfully to 'object_detection_pipeline.yaml'")