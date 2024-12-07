import importlib.util
import os

import pytest

EXAMPLES_DIR = "examples"

# Collect all Python files in the examples directory
example_files = [file for file in os.listdir(EXAMPLES_DIR) if file.endswith(".py")]


@pytest.mark.parametrize("script", example_files)
def test_example_scripts(script):
    script_path = os.path.join(EXAMPLES_DIR, script)
    module_name = script[:-3]  # Remove the .py extension

    # Dynamically import the script
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # Execute the module
    except Exception as e:
        pytest.fail(f"Script {script} raised an exception: {e}")
