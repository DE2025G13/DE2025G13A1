import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Artifact, Model, OutputPath

@dsl.container_component
def preprocess_data(
    data_bucket_name: str,
    raw_data_path: str,
    processed_data: dsl.Output[dsl.Dataset],
):
    """Initial component to load, split, and process the raw data."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/data-preprocessor:latest',
        command=["python3", "component.py"],
        args=[
            "--data-bucket-name", data_bucket_name,
            "--raw-data-path", raw_data_path,
            "--processed-data-path", processed_data.path,
        ]
    )

@dsl.container_component
def train_model(
    image_url: str,
    processed_data: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
):
    """A generic training component that takes a Docker image URL."""
    return dsl.ContainerSpec(
        image=image_url,
        command=["python3", "component.py"],
        args=[
            "--processed-data-path", processed_data.path,
            "--model-artifact-path", model.path,
        ]
    )

@dsl.container_component
def evaluate_models(
    processed_data: dsl.Input[dsl.Dataset],
    decision_tree_model: dsl.Input[dsl.Model],
    linear_regression_model: dsl.Input[dsl.Model],
    logistic_regression_model: dsl.Input[dsl.Model],
    model_bucket_name: str,
    prod_model_blob: str,
    decision: OutputPath(str),
):
    """Evaluates models and PRINTS the GCS URI of the best model or 'keep_old' to stdout."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-evaluator:latest',
        command=["bash", "-c"],
        args=[
            f"python component.py "
            f"--processed-data-path {processed_data.path} "
            f"--decision-tree-model-path {decision_tree_model.path} "
            f"--linear-regression-model-path {linear_regression_model.path} "
            f"--logistic-regression-model-path {logistic_regression_model.path} "
            f"--model-bucket-name {model_bucket_name} "
            f"--prod-model-blob {prod_model_blob} "
            f"> {decision}"
        ]
    )

@dsl.container_component
def trigger_cd_pipeline(
    project_id: str,
    trigger_id: str,
    new_model_uri: str,
):
    """Triggers the Cloud Build deployment pipeline."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/trigger-cd:latest',
        command=["python3", "component.py"],
        args=[
            "--project-id", project_id,
            "--trigger-id", trigger_id,
            "--new-model-uri", new_model_uri,
            "--best-model-name", "best_model"
        ]
    )

@dsl.pipeline(
    name='wine-quality-training-pipeline',
    description='Trains, evaluates, and triggers deployment for a wine quality model.'
)
def wine_quality_pipeline(
    project_id: str = "data-engineering-vm",
    data_bucket: str = "yannick-wine-data",
    model_bucket: str = "yannick-wine-models",
    pipeline_root: str = "gs://yannick-wine-pipeline-root",
    cd_trigger_id: str = "deploy-wine-app-trigger"
):
    preprocess_task = preprocess_data(
        data_bucket_name=data_bucket,
        raw_data_path="raw/WineQT.csv"
    )

    train_dt_task = train_model(
        image_url='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-dt:latest',
        processed_data=preprocess_task.outputs["processed_data"]
    )
    train_lr_task = train_model(
        image_url='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-lr:latest',
        processed_data=preprocess_task.outputs["processed_data"]
    )
    train_logr_task = train_model(
        image_url='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-logr:latest',
        processed_data=preprocess_task.outputs["processed_data"]
    )
    
    evaluation_task = evaluate_models(
        processed_data=preprocess_task.outputs["processed_data"],
        decision_tree_model=train_dt_task.outputs["model"],
        linear_regression_model=train_lr_task.outputs["model"],
        logistic_regression_model=train_logr_task.outputs["model"],
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib"
    )
    
    evaluation_task.set_caching_options(enable_caching=False)

    with dsl.Condition(evaluation_task.outputs["decision"] != "keep_old", name="if-new-model-is-better"):
        trigger_cd_pipeline(
            project_id=project_id,
            trigger_id=cd_trigger_id,
            new_model_uri=evaluation_task.outputs["decision"],
        )

if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler(mode=dsl.PipelineExecutionMode.V1_LEGACY).compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.json'
    )
    print("\nPipeline compiled successfully to wine_quality_pipeline.json using V1_LEGACY mode.\n")