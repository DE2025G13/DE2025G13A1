import kfp
from kfp import dsl
from kfp.dsl import Output, Artifact

@dsl.container_component
def auto_labeler(
    raw_bucket_name: str,
    processed_bucket_name: str,
    processed_prefix: str,
    labeled_dataset: Output[Artifact]
):
    return dsl.ContainerSpec(
        image='us-central1-docker.pkg.dev/data-engineering-vm/ml-services/auto-labeler:latest',
        command=[ "python3", "component.py", "--raw-bucket-name", raw_bucket_name, "--processed-bucket-name", processed_bucket_name, "--processed-prefix", processed_prefix, ]
    )

@dsl.container_component
def dataset_splitter(
    bucket_name: str,
    prefix: str,
    data_yaml_artifact: Output[Artifact]
):
    return dsl.ContainerSpec(
        image='us-central1-docker.pkg.dev/data-engineering-vm/ml-services/dataset-splitter:latest',
        command=[ "python3", "component.py", "--bucket-name", bucket_name, "--prefix", prefix, ]
    )

@dsl.container_component
def yolo_trainer(
    data_yaml_uri: str,
    output_bucket_name: str,
    output_blob_name: str,
    model_artifact: Output[Artifact]
):
    return dsl.ContainerSpec(
        image='us-central1-docker.pkg.dev/data-engineering-vm/ml-services/yolo-trainer:latest',
        command=[ "python", "component.py", "--data-yaml-uri", data_yaml_uri, "--output-bucket-name", output_bucket_name, "--output-blob-name", output_blob_name ]
    ).set_gpu_limit(1).set_accelerator_type('NVIDIA_TESLA_T4')

@dsl.container_component
def model_evaluator(
    data_yaml_uri: str,
    new_model_uri: str,
    prod_model_uri: str
) -> str:
    return dsl.ContainerSpec(
        image='us-central1-docker.pkg.dev/data-engineering-vm/ml-services/model-evaluator:latest',
        command=[ "python", "component.py", "--data-yaml-uri", data_yaml_uri, "--new-model-uri", new_model_uri, "--prod-model-uri", prod_model_uri ]
    ).set_gpu_limit(1).set_accelerator_type('NVIDIA_TESLA_T4')

@dsl.container_component
def trigger_cd_pipeline( webhook_url: str, new_model_uri: str ):
    return dsl.ContainerSpec(
        image='google/cloud-sdk:slim',
        command=[ "bash", "-c", f'curl -X POST -H "Content-Type: application/json" -d \'{{"substitutions":{{"_NEW_MODEL_URI":"{new_model_uri}"}}}}\' "{webhook_url}"' ]
    )

@dsl.pipeline( name='yolo-glasses-detection-pipeline' )
def object_detection_pipeline(
    project_id: str = "data-engineering-vm",
    pipeline_root: str = "gs://glasses-temp/pipeline-root",
    raw_bucket: str = "glasses-data",
    processed_bucket: str = "glasses-data",
    model_bucket: str = "glasses-model",
    cd_webhook_url: str = "https://cloudbuild.googleapis.com/v1/projects/data-engineering-vm/triggers/deploy-glasses-app-webhook:webhook?key=AIzaSyC_DDAlTTaytzyeYLxLFXLIZhu1sDe8Crc"
):
    processed_prefix = f"labeled_data/run-{{{{$.pipeline_job_name}}}}"
    new_model_blob = f"candidate_models/{{{{$.pipeline_job_name}}}}/best.pt"

    labeling_task = auto_labeler(
        raw_bucket_name=raw_bucket,
        processed_bucket_name=processed_bucket,
        processed_prefix=processed_prefix
    )

    splitting_task = dataset_splitter(
        bucket_name=processed_bucket,
        prefix=processed_prefix
    ).after(labeling_task)
    
    data_yaml_gcs_uri = f"gs://{processed_bucket}/{processed_prefix}/data.yaml"

    training_task = yolo_trainer(
        data_yaml_uri=data_yaml_gcs_uri,
        output_bucket_name=model_bucket,
        output_blob_name=new_model_blob
    ).after(splitting_task)
    
    new_model_gcs_uri = f"gs://{model_bucket}/{new_model_blob}"

    evaluation_task = model_evaluator(
        data_yaml_uri=data_yaml_gcs_uri,
        new_model_uri=new_model_gcs_uri,
        prod_model_uri=f"gs://{model_bucket}/production_model/best.pt"
    ).after(training_task)

    with dsl.Condition(evaluation_task.output == "deploy", name="deploy-trigger"):
        trigger_cd_pipeline(
            webhook_url=cd_webhook_url,
            new_model_uri=new_model_gcs_uri
        )

if __name__ == '__main__':
    from kfp.compiler import Compiler
    Compiler().compile(
        pipeline_func=object_detection_pipeline,
        package_path='object_detection_pipeline.yaml'
    )