import kfp
from kfp import dsl
from kfp.dsl import Output, Artifact, OutputPath

@dsl.container_component
def auto_labeler(
    raw_bucket_name: str,
    processed_bucket_name: str,
    processed_prefix: str,
    labeled_dataset: Output[Artifact]
):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/braided-voyage-472209-n6/ml-services/auto-labeler:latest',
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
        image='europe-west4-docker.pkg.dev/braided-voyage-472209-n6/ml-services/dataset-splitter:latest',
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
    model_artifact: Output[Artifact]
):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/braided-voyage-472209-n6/ml-services/yolo-trainer:latest',
        command=[
            "python", "component.py",
            "--data-yaml-uri", data_yaml_uri,
            "--output-bucket-name", output_bucket_name,
            "--output-blob-name", output_blob_name
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
        image='europe-west4-docker.pkg.dev/braided-voyage-472209-n6/ml-services/model-evaluator:latest',
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
    return dsl.ContainerSpec(
        image='google/cloud-sdk:slim',
        command=[
            "bash", "-c",
            f"""
            set -e
            ACCESS_TOKEN=$(gcloud auth print-access-token)
            curl -X POST -H "Authorization: Bearer $ACCESS_TOKEN" \
                 -H "Content-Type: application/json" \
                 "https://cloudbuild.googleapis.com/v1/projects/{project_id}/triggers/{trigger_id}:run" \
                 -d '{{"substitutions":{{"_NEW_MODEL_URI":"{new_model_uri}"}}}}'
            """
        ]
    )

@dsl.pipeline(
    name='yolo-glasses-detection-pipeline',
    description='End-to-end pipeline for training and deploying glasses detection model'
)
def object_detection_pipeline(
    project_id: str = "braided-voyage-472209-n6",
    pipeline_root: str = "gs://glasses-temp_daan/pipeline-root",
    raw_bucket: str = "glasses-data_daan",
    processed_bucket: str = "glasses-data_daan",
    model_bucket: str = "glasses-model_daan",
    cd_trigger_id: str = "deploy-glasses-app"
):
    # Dynamic paths using pipeline context
    processed_prefix = "labeled_data/run-{{$.pipeline_job_name}}"
    new_model_blob = "candidate_models/{{$.pipeline_job_name}}/best.pt"

    # Step 1: Auto-label the raw images
    labeling_task = auto_labeler(
        raw_bucket_name=raw_bucket,
        processed_bucket_name=processed_bucket,
        processed_prefix=processed_prefix
    )

    # Step 2: Split dataset into train/val/test
    splitting_task = dataset_splitter(
        bucket_name=processed_bucket,
        prefix=processed_prefix
    ).after(labeling_task)
    
    # Construct the data.yaml GCS path
    data_yaml_gcs_uri = f"gs://{processed_bucket}/{processed_prefix}/data.yaml"

    # Step 3: Train YOLOv8 model
    training_task = yolo_trainer(
        data_yaml_uri=data_yaml_gcs_uri,
        output_bucket_name=model_bucket,
        output_blob_name=new_model_blob
    ).after(splitting_task)
    
    # Configure GPU for training task
    training_task.set_gpu_limit(1)
    training_task.set_accelerator_type('NVIDIA_TESLA_T4')
    
    # Construct the new model GCS path
    new_model_gcs_uri = f"gs://{model_bucket}/{new_model_blob}"

    # Step 4: Evaluate new model against production model
    evaluation_task = model_evaluator(
        data_yaml_uri=data_yaml_gcs_uri,
        new_model_uri=new_model_gcs_uri,
        prod_model_uri=f"gs://{model_bucket}/production_model/best.pt"
    ).after(training_task)
    
    # Configure GPU for evaluation task
    evaluation_task.set_gpu_limit(1)
    evaluation_task.set_accelerator_type('NVIDIA_TESLA_T4')

    # Step 5: Conditionally trigger CD pipeline if new model is better
    with dsl.Condition(
        evaluation_task.outputs['decision'] == "deploy",
        name="deploy-new-model"
    ):
        trigger_cd_pipeline(
            project_id=project_id,
            trigger_id=cd_trigger_id,
            new_model_uri=new_model_gcs_uri
        )

if __name__ == '__main__':
    from kfp.compiler import Compiler
    Compiler().compile(
        pipeline_func=object_detection_pipeline,
        package_path='object_detection_pipeline.yaml'
    )
    print("Pipeline compiled successfully to 'object_detection_pipeline.yaml'")