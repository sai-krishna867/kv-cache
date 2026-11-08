import pytest
from pydantic import ValidationError

from kvstudy.config import ExperimentConfig, config_hash


def test_valid_config(config_dict):
    cfg = ExperimentConfig.model_validate(config_dict)
    assert cfg.methods[0].id == "bf16"
    assert len(config_hash(cfg)) == 64


def test_config_hash_is_stable(config_dict):
    assert config_hash(ExperimentConfig.model_validate(config_dict)) == config_hash(
        ExperimentConfig.model_validate(config_dict)
    )


def test_unknown_top_level_field_rejected(config_dict):
    config_dict["extra"] = 1
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


def test_unknown_nested_field_rejected(config_dict):
    config_dict["protocol"]["protected_window"] = {"tokens": 512}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


def test_systems_real_rejected_not_substituted(config_dict):
    config_dict["execution"]["mode"] = "systems_real"
    with pytest.raises(ValidationError, match="milestone E"):
        ExperimentConfig.model_validate(config_dict)


def test_unimplemented_method_kind_rejected(config_dict):
    config_dict["methods"].append({"id": "u4", "kind": "uniform", "key_bits": 4, "value_bits": 4})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


def test_duplicate_method_ids_rejected(config_dict):
    config_dict["methods"].append({"id": "bf16", "kind": "identity"})
    with pytest.raises(ValidationError, match="duplicate method ids"):
        ExperimentConfig.model_validate(config_dict)


def test_unsafe_method_id_rejected(config_dict):
    config_dict["methods"][0]["id"] = "../escape"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_non_bf16_weights_rejected(config_dict, dtype):
    config_dict["model"]["weight_dtype"] = dtype
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


@pytest.mark.parametrize("revision", [None, "main", "abc123"])
def test_revision_must_be_pinned_commit(config_dict, revision):
    if revision is None:
        del config_dict["model"]["revision"]
    else:
        config_dict["model"]["revision"] = revision
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


def test_unknown_benchmark_rejected(config_dict):
    config_dict["benchmarks"][0]["name"] = "longbench"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


WINDOW = {"tokens": 512, "precision_bits": 8, "overflow": "demote_oldest"}
UNIFORM = {"id": "u4", "kind": "uniform", "key_bits": 4, "value_bits": 4, "group_size": 64, "axis": "per_token"}


def test_uniform_with_window_accepted(config_dict):
    config_dict["protocol"]["protected_window"] = WINDOW
    config_dict["methods"].append(dict(UNIFORM))
    cfg = ExperimentConfig.model_validate(config_dict)
    assert cfg.methods[1].key_bits == 4


def test_compressed_method_requires_protected_window(config_dict):
    config_dict["methods"].append(dict(UNIFORM))
    with pytest.raises(ValidationError, match="protected_window"):
        ExperimentConfig.model_validate(config_dict)


def test_window_below_method_precision_rejected(config_dict):
    config_dict["protocol"]["protected_window"] = {**WINDOW, "precision_bits": 4}
    config_dict["methods"].append({**UNIFORM, "key_bits": 8})
    with pytest.raises(ValidationError, match="below the method"):
        ExperimentConfig.model_validate(config_dict)


@pytest.mark.parametrize("field,value", [("key_bits", 7), ("value_bits", 16), ("group_size", 48)])
def test_unsupported_quantization_values_rejected(config_dict, field, value):
    config_dict["protocol"]["protected_window"] = WINDOW
    config_dict["methods"].append({**UNIFORM, field: value})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


def test_per_channel_axis_rejected(config_dict):
    config_dict["protocol"]["protected_window"] = WINDOW
    config_dict["methods"].append({**UNIFORM, "axis": "per_channel"})
    with pytest.raises(ValidationError, match="not implemented"):
        ExperimentConfig.model_validate(config_dict)


def test_unknown_window_overflow_policy_rejected(config_dict):
    config_dict["protocol"]["protected_window"] = {**WINDOW, "overflow": "evict"}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


SELECTIVE = {
    "id": "s82", "kind": "selective", "selector": "quantization_residual", "high_bits": 8, "low_bits": 2,
    "high_fraction": 0.165, "group_size": 64, "axis": "per_token",
}  # fmt: skip


def test_selective_accepted(config_dict):
    config_dict["protocol"]["protected_window"] = WINDOW
    config_dict["methods"] += [SELECTIVE, {**SELECTIVE, "id": "rnd", "selector": "random", "seed": 0}]
    cfg = ExperimentConfig.model_validate(config_dict)
    assert cfg.methods[1].high_fraction == 0.165 and cfg.methods[2].seed == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"selector": "attention"}, "selector"),
        ({"selector": "random"}, "requires seed"),
        ({"selector": "sinks_recency"}, "requires sink_tokens"),
        ({"seed": 3}, "does not apply"),
        ({"high_bits": 2}, "must exceed"),
        ({"high_fraction": 1.0}, "high_fraction"),
        ({"high_fraction": 0}, "high_fraction"),
        ({"axis": "per_channel"}, "not implemented"),
    ],
)
def test_selective_rejections(config_dict, change, match):
    config_dict["protocol"]["protected_window"] = WINDOW
    config_dict["methods"].append({**SELECTIVE, **change})
    with pytest.raises(ValidationError, match=match):
        ExperimentConfig.model_validate(config_dict)


def test_window_below_selective_high_bits_rejected(config_dict):
    config_dict["protocol"]["protected_window"] = {**WINDOW, "precision_bits": 4}
    config_dict["methods"].append({**SELECTIVE, "high_bits": 8, "low_bits": 2})
    with pytest.raises(ValidationError, match="below the method"):
        ExperimentConfig.model_validate(config_dict)
