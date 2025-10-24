import kfp
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Artifact, Model, Dataset, OutputPath

@dsl.container_component
def data_ingestion(
    bucket_name: str,
    blob_name: str,
    raw_data: Output[Artifact],
):
    """Downloads data from GCS and makes it a KFP artifact."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/data-ingestion:latest',
        command=["python3", "component.py", "--bucket-name", bucket_name, "--blob-name", blob_name, "--output-path", raw_data.path]
    )

@dsl.container_component
def train_test_splitter(
    input_data: Input[Artifact],
    training_data: Output[Dataset],
    testing_data: Output[Dataset],
):
    """Splits raw data into training and testing sets."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/train-test-splitter:latest',
        command=[
            "python3", "component.py",
            "--input-data-path", input_data.path,
            "--training-data-path", training_data.path,
            "--testing-data-path", testing_data.path,
        ]
    )

@dsl.container_component
def train_decision_tree(training_data: Input[Dataset], model: Output[Model]):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-dt:latest',
        command=["python3", "component.py", "--training_data_path", training_data.path, "--model_artifact_path", model.path]
    )

@dsl.container_component
def train_linear_regression(training_data: Input[Dataset], model: Output[Model]):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-lr:latest',
        command=["python3", "component.py", "--training_data_path", training_data.path, "--model_artifact_path", model.path]
    )

@dsl.container_component
def train_logistic_regression(training_data: Input[Dataset], model: Output[Model]):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-trainer-logr:latest',
        command=["python3", "component.py", "--training_data_path", training_data.path, "--model_artifact_path", model.path]
    )

@dsl.container_component
def model_evaluator(
    testing_data: Input[Dataset],
    decision_tree_model: Input[Model],
    linear_regression_model: Input[Model],
    logistic_regression_model: Input[Model],
    model_bucket_name: str,
    prod_model_blob: str,
    decision: OutputPath(str),
):
    """Finds the best candidate model and compares it to production."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-evaluator:latest',
        command=[
            "bash", "-c",
            f"python component.py "
            f"--testing_data_path {testing_data.path} "
            f"--decision_tree_model_path {decision_tree_model.path} "
            f"--linear_regression_model_path {linear_regression_model.path} "
            f"--logistic_regression_model_path {logistic_regression_model.path} "
            f"--model_bucket_name {model_bucket_name} "
            f"--prod_model_blob {prod_model_blob} "
            f"> {decision}"
        ]
    )

@dsl.container_component
def trigger_cd_pipeline(project_id: str, trigger_id: str, new_model_uri: str):
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/trigger-cd:latest',
        command=["python3", "component.py", "--project-id", project_id, "--trigger-id", trigger_id, "--new-model-uri", new_model_uri, "--best-model-name", "best_model"]
    )

@dsl.pipeline(
    name='wine-quality-elaborate-pipeline',
    description='A robust pipeline that ingests, splits, trains, and evaluates models.'
)
def wine_quality_pipeline(
    project_id: str = "data-engineering-vm",
    data_bucket: str = "yannick-wine-data",
    model_bucket: str = "yannick-wine-models",
    cd_trigger_id: str = "deploy-wine-app-trigger"
):
    ingestion_task = data_ingestion(
        bucket_name=data_bucket,
        blob_name="raw/WineQT.csv"
    )

    split_task = train_test_splitter(
        input_data=ingestion_task.outputs["raw_data"]
    )

    dt_task = train_decision_tree(training_data=split_task.outputs["training_data"])
    lr_task = train_linear_regression(training_data=split_task.outputs["training_data"])
    logr_task = train_logistic_regression(training_data=split_task.outputs["training_data"])

    eval_task = model_evaluator(
        testing_data=split_task.outputs["testing_data"],
        decision_tree_model=dt_task.outputs["model"],
        linear_regression_model=lr_task.outputs["model"],
        logistic_regression_model=logr_task.outputs["model"],
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib"
    )
    eval_task.set_caching_options(enable_caching=False)

    with dsl.Condition(eval_task.outputs["decision"] != "keep_old", name="if-new-model-is-better"):
        with dsl.Condition(eval_task.outputs["decision"] == "decision_tree", name="deploy-decision-tree"):
            trigger_cd_pipeline(project_id=project_id, trigger_id=cd_trigger_id, new_model_uri=dt_task.outputs["model"].uri)
        with dsl.Condition(eval_task.outputs["decision"] == "linear_regression", name="deploy-linear-regression"):
            trigger_cd_pipeline(project_id=project_id, trigger_id=cd_trigger_id, new_model_uri=lr_task.outputs["model"].uri)
        with dsl.Condition(eval_task.outputs["decision"] == "logistic_regression", name="deploy-logistic-regression"):
            trigger_cd_pipeline(project_id=project_id, trigger_id=cd_trigger_id, new_model_uri=logr_task.outputs["model"].uri)

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("\nPipeline compiled successfully to wine_quality_pipeline.yaml\n")
