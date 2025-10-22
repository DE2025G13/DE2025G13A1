import kfp
from kfp import dsl

@dsl.container_component
def yolo_trainer_debug_test():
    """A minimal component that just runs the yolo-trainer:debug image."""
    return dsl.ContainerSpec(
        image='europe-west4-docker.pkg.dev/data-engineering-vm/ml-services/yolo-trainer:debug',
    )

@dsl.pipeline(
    name='yolo-trainer-debug-startup-test',
    description='A barebones pipeline to test if the yolo-trainer:debug image can start on a GPU worker.'
)
def debug_pipeline(
    project_id: str = "data-engineering-vm",
    pipeline_root: str = "gs://glasses-temp/pipeline-root"
):
    """This pipeline runs only one step: our debug test."""
    
    # Step 1: Instantiate the debug component.
    debug_task = yolo_trainer_debug_test()

    # Step 2: Assign a GPU to the task. This is the critical part of the test.
    debug_task.set_gpu_limit(1)
    debug_task.set_accelerator_type('NVIDIA_TESLA_T4')

if __name__ == '__main__':
    from kfp.compiler import Compiler
    Compiler().compile(
        pipeline_func=debug_pipeline,
        package_path='object_detection_pipeline.yaml'
    )
    print("Barebones Debug Test Pipeline compiled successfully to 'object_detection_pipeline.yaml'")