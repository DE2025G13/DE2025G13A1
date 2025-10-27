import os
import pytest
import importlib.util

CD_TRIGGER_PATH = "components/trigger-cd/component.py"

def import_trigger():
    spec = importlib.util.spec_from_file_location("trigger_cd", CD_TRIGGER_PATH)
    if not spec:
        pytest.skip(f"Module not found: {CD_TRIGGER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'trigger_cloud_build'):
        pytest.fail(f"Function 'trigger_cloud_build' not found")
    return module.trigger_cloud_build

trigger_cloud_build = import_trigger()

def test_trigger_cd_function_exists():
    assert callable(trigger_cloud_build), "trigger_cloud_build should be callable"

def test_trigger_cd_params():
    import inspect
    sig = inspect.signature(trigger_cloud_build)
    params = list(sig.parameters.keys())
    assert "project_id" in params, "Missing project_id parameter"
    assert "trigger_id" in params, "Missing trigger_id parameter"