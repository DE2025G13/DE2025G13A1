import kfp
from kfp import dsl
from kfp import compiler

@dsl.container_component
def preprocess_data(
    data_bucket_name: str,
    raw_data_path: str,
    processed_prefix: str,
):
    """Writes processed data to a specified GCS prefix."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/data-preprocessor:latest',
        command=[ "python3", "component.py", "--data-bucket-name", data_bucket_name, "--raw-data-path", raw_data_path, "--processed-prefix", processed_prefix ]
    )

@dsl.container_component
def train_model(
    image_url: str,
    processed_data_uri: str,
    output_model_uri: str,
):
    """Reads data from a URI and writes a model to a URI."""
    return dsl.ContainerSpec(
        image=image_url,
        command=[ "python3", "component.py", "--processed-data-uri", processed_data_uri, "--output-model-uri", output_model_uri ]
    )

@dsl.container_component
def evaluate_models(
    processed_data_uri: str,
    decision_tree_model_uri: str,
    linear_regression_model_uri: str,
    logistic_regression_model_uri: str,
    model_bucket_name: str,
    prod_model_blob: str,
    decision: dsl.OutputPath(str),
):
    """Evaluates models and prints the decision."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-evaluator:latest',
        command=[
            "bash", "-c",
            f"python component.py "
            f"--processed-data-uri {processed_data_uri} "
            f"--decision-tree-model-uri {decision_tree_model_uri} "
            f"--linear-regression-model-uri {linear_regression_model_uri} "
            f"--logistic-regression-model-uri {logistic_regression_model_uri} "
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
        command=[ "python3", "component.py", "--project-id", project_id, "--trigger-id", trigger_id, "--new-model-uri", new_model_uri, "--best-model-name", "best_model" ]
    )

@dsl.pipeline(
    name='wine-quality-training-pipeline',
    description='Trains, evaluates, and triggers deployment for a wine quality model.'
)
def wine_quality_pipeline(
    project_id: str = "data-engineering-vm",
    data_bucket: str = "yannick-wine-data",
    model_bucket: str = "yannick-wine-models",
    cd_trigger_id: str = "deploy-wine-app-trigger"
):
    processed_prefix = f"processed_data/run-{{{{$.pipeline_job_name}}}}"
    processed_data_uri = f"gs://{data_bucket}/{processed_prefix}"

    # Step 1: Preprocess the data. It writes to the URI we just defined.
    preprocess_task = preprocess_data(
        data_bucket_name=data_bucket,
        raw_data_path="raw/WineQT.csv",
        processed_prefix=processed_prefix
    )

    dt_model_uri = f"gs://{model_bucket}/candidate_models/{{{{$.pipeline_job_name}}}}/decision-tree/model.joblib"
    lr_model_uri = f"gs://{model_bucket}/candidate_models/{{{{$.pipeline_job_name}}}}/linear-regression/model.joblib"
    logr_model_uri = f"gs://{model_bucket}/candidate_models/{{{{$.pipeline_job_name}}}}/logistic-regression/model.joblib"

    # Step 2: Train models. They read and write from the URIs we defined.
    train_dt_task = train_model(image_url='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-dt:latest', processed_data_uri=processed_data_uri, output_model_uri=dt_model_uri).after(preprocess_task)
    train_lr_task = train_model(image_url='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-lr:latest', processed_data_uri=processed_data_uri, output_model_uri=lr_model_uri).after(preprocess_task)
    train_logr_task = train_model(image_url='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-logr:latest', processed_data_uri=processed_data_uri, output_model_uri=logr_model_uri).after(preprocess_task)
    
    # Step 3: Evaluate. Pass all paths as simple strings.
    evaluation_task = evaluate_models(
        processed_data_uri=processed_data_uri,
        decision_tree_model_uri=dt_model_uri,
        linear_regression_model_uri=lr_model_uri,
        logistic_regression_model_uri=logr_model_uri,
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib"
    ).after(train_dt_task, train_lr_task, train_logr_task)
    
    evaluation_task.set_caching_options(enable_caching=False)
    
    with dsl.Condition(evaluation_task.outputs["decision"] != "keep_old", name="if-new-model-is-better"):
        trigger_cd_pipeline(project_id=project_id, trigger_id=cd_trigger_id, new_model_uri=evaluation_task.outputs["decision"])

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("\nPipeline compiled successfully to wine_quality_pipeline.yaml\n")