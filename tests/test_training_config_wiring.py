import ast
from pathlib import Path

import yaml


def test_compile_model_accepts_configurable_loss_weights():
    source = Path("src/cardiofit/models/resnet_se_lstm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    compile_def = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compile_model"
    )

    arg_names = [arg.arg for arg in compile_def.args.args]

    assert "loss_weights" in arg_names


def test_training_script_passes_config_loss_weights_to_compile_model():
    source = Path("scripts/train_multimodal.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "compile_model"
    ]

    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert "loss_weights" in keyword_names


def test_scg_rhc_config_disables_placeholder_vo2_loss():
    cfg = yaml.safe_load(Path("configs/scg_rhc_windows5090d.yaml").read_text(encoding="utf-8"))

    assert cfg["training"]["loss_weights"] == {
        "co_output": 1.0,
        "vo2_output": 0.0,
    }
