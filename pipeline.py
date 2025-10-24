import kfp
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model

# Component: Preprocess Data
# Uses the modern @dsl.component decorator for clarity and stability.
@dsl.component(
    base_image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/data-preprocessor:latest',
)
def preprocess_data(
    data_bucket_name: str,
    raw_data_path: str,
    processed_data: Output[Dataset],
):
    pass

# Component: Generic Model Trainer
@dsl.component
def train_model(
    image_url: str,
    processed_data: Input[Dataset],
    model: Output[Model],
):
    return dsl.ContainerSpec(
        image=image_url,
        command=[
            "python3",
            "component.py",
            "--processed-data-path",
            processed_data.path,
            "--model-artifact-path",
            model.path,
        ]
    )

# Component: Evaluate Models
@dsl.component
def evaluate_models(
    processed_data: Input[Dataset],
    decision_tree_model: Input[Model],
    linear_regression_model: Input[Model],
    logistic_regression_model: Input[Model],
    model_bucket_name: str,
    prod_model_blob: str,
) -> str:
    """Evaluates models and returns 'deploy' or 'keep_old'."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/model-evaluator:latest',
        command=[
            "python3", "component.py",
            "--processed-data-path", processed_data.path,
            "--decision-tree-model-path", decision_tree_model.path,
            "--linear-regression-model-path", linear_regression_model.path,
            "--logistic-regression-model-path", logistic_regression_model.path,
            "--model-bucket-name", model_bucket_name,
            "--prod-model-blob", prod_model_blob,
            "--output-path", dsl.OutputPath(str),
        ]
    )

# Component: Trigger Cloud Build CD Pipeline
@dsl.component(
    base_image='europe-west4-docker.pkg.dev/data-engineering-vm/yannick-wine-repo/trigger-cd:latest',
)
def trigger_cd_pipeline(
    project_id: str,
    trigger_id: str,
    new_model_uri: str,
):
    """Triggers the Cloud Build deployment pipeline."""
    pass

# The main pipeline definition
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
    # Step 1: Preprocess the data
    preprocess_task = preprocess_data(
        data_bucket_name=data_bucket,
        raw_data_path="raw/WineQT.csv"
    )

    # Step 2: Train all three models in parallel
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
    
    # Step 3: Evaluate the models after they are all trained
    evaluation_task = evaluate_models(
        processed_data=preprocess_task.outputs["processed_data"],
        decision_tree_model=train_dt_task.outputs["model"],
        linear_regression_model=train_lr_task.outputs["model"],
        logistic_regression_model=train_logr_task.outputs["model"],
        model_bucket_name=model_bucket,
        prod_model_blob="production_model/model.joblib"
    )
    
    evaluation_task.set_caching_options(enable_caching=False)

    # Step 4: Conditionally trigger the deployment pipeline
    with dsl.If(evaluation_task.output != "keep_old", name="if-new-model-is-better"):
        trigger_cd_pipeline(
            project_id=project_id,
            trigger_id=cd_trigger_id,
            new_model_uri=evaluation_task.output,
        )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=wine_quality_pipeline,
        package_path='wine_quality_pipeline.yaml'
    )
    print("\nPipeline compiled successfully to wine_quality_pipeline.yaml\n")