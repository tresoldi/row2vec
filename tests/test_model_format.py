"""The .r2v model file: faithful round trips, and loading that never runs code."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import skops.io as sio

import row2vec
from row2vec import ModelFormatError, inspect_model, load_model, save_model
from row2vec import serialization as ser


def messy_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Numeric and categorical columns with gaps and a high-cardinality column."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(10, 3, size=n),
            "small": rng.choice(["x", "y", "z"], size=n),
            "big": [f"code_{i % 130:03d}" for i in range(n)],
            "label": rng.choice(["u", "v", "w"], size=n),
        }
    )
    frame.loc[rng.choice(n, 20, replace=False), "num_a"] = np.nan
    frame.loc[rng.choice(n, 20, replace=False), "small"] = None
    return frame


def fit(mode: str, frame: pd.DataFrame, **kwargs: Any) -> row2vec.Row2VecModel:
    _, model = row2vec.learn_embedding_with_model(
        frame, mode=mode, embedding_dim=2, enable_logging=False, **kwargs
    )
    return model


def rewrite(
    path: Path, replace: dict[str, bytes] | None = None, drop: tuple[str, ...] = ()
) -> None:
    """Rewrite an archive with changed members *and a matching manifest*.

    This is what an attacker who understands the format would do, so a test that
    passes here is not relying on the checksum.
    """
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members.update(replace or {})
    for name in drop:
        members.pop(name, None)
    manifest = json.loads(members["manifest.json"])
    manifest["members"] = {
        name: hashlib.sha256(blob).hexdigest()
        for name, blob in members.items()
        if name != "manifest.json"
    }
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in members.items():
            archive.writestr(name, blob)


class Gadget:
    """A type that is not on the allow-list."""

    def __init__(self) -> None:
        self.payload = 1


# --------------------------------------------------------------------------- #
# Faithful round trips
# --------------------------------------------------------------------------- #

CLASSICAL = [("pca", {}), ("umap", {"n_neighbors": 8})]
NEURAL = [
    ("unsupervised", {"max_epochs": 2}),
    ("target", {"max_epochs": 2, "reference_column": "label"}),
    ("contrastive", {"max_epochs": 2, "auto_pairs": "neighbors"}),
]


@pytest.mark.parametrize(("mode", "kwargs"), CLASSICAL)
def test_classical_round_trip_is_exact(mode: str, kwargs: dict[str, Any], tmp_path: Path) -> None:
    frame = messy_frame()
    model = fit(mode, frame, **kwargs)
    new = messy_frame(40, seed=7)

    path = save_model(model, tmp_path / "m")
    restored = load_model(path)

    np.testing.assert_allclose(
        restored.predict(new, validate_schema=False).to_numpy(),
        model.predict(new, validate_schema=False).to_numpy(),
        atol=1e-8,
    )


@pytest.mark.neural
@pytest.mark.parametrize(("mode", "kwargs"), NEURAL)
def test_neural_round_trip_is_exact(mode: str, kwargs: dict[str, Any], tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    frame = messy_frame()
    model = fit(mode, frame, **kwargs)
    new = messy_frame(40, seed=7)

    restored = load_model(save_model(model, tmp_path / "m"))

    np.testing.assert_allclose(
        restored.predict(new, validate_schema=False).to_numpy(),
        model.predict(new, validate_schema=False).to_numpy(),
        atol=1e-6,
    )
    assert restored.config == model.config


@pytest.mark.neural
def test_target_encoded_column_round_trips_and_carries_no_row_data(tmp_path: Path) -> None:
    """Target encoding stores cross-fitted values per training row; they must not be saved."""
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, size=n)
    frame = pd.DataFrame(
        {
            "mid": [f"m{(i % 50) + 25 * label:02d}" for i, label in enumerate(y)],
            "x": rng.normal(size=n),
            "label": y,
        }
    )
    model = fit("target", frame, reference_column="label", max_epochs=2)
    pre = model.preprocessor_
    assert pre is not None
    encoders = [
        step
        for _, transformer, _ in pre.transformers_
        for _, step in getattr(transformer, "steps", [])
        if hasattr(step, "cv_encodings_")
    ]
    assert encoders, "expected a target-encoded column in the preprocessor"
    assert all(encoder.cv_encodings_ == {} for encoder in encoders)

    restored = load_model(save_model(model, tmp_path / "m"))
    np.testing.assert_allclose(
        restored.predict(frame, validate_schema=False).to_numpy(),
        model.predict(frame, validate_schema=False).to_numpy(),
        atol=1e-8,
    )


def test_tsne_model_can_be_saved_and_still_refuses_to_transform(tmp_path: Path) -> None:
    model = fit("tsne", messy_frame(), perplexity=10)
    restored = load_model(save_model(model, tmp_path / "m"))
    with pytest.raises(NotImplementedError, match="t-SNE"):
        restored.transform(messy_frame(40))


def test_integer_column_names_survive(tmp_path: Path) -> None:
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 4)), columns=[0, 1, 2, 3])
    model = fit("pca", frame)

    restored = load_model(save_model(model, tmp_path / "m"))

    assert restored.validate_input_schema(frame, strict=False)
    np.testing.assert_allclose(restored.predict(frame).to_numpy(), model.predict(frame).to_numpy())


def test_metadata_and_state_survive(tmp_path: Path) -> None:
    frame = messy_frame()
    model = fit("pca", frame)
    model.metadata = ser.describe_model(model)

    restored = load_model(save_model(model, tmp_path / "m"))

    assert restored.metadata is not None
    assert restored.metadata.to_dict()["data_shape"] == model.metadata.to_dict()["data_shape"]
    assert restored.training_columns_ == model.training_columns_
    assert restored.training_shape_ == model.training_shape_
    assert restored.config == model.config


def test_suffix_is_optional(tmp_path: Path) -> None:
    model = fit("pca", messy_frame())
    path = save_model(model, tmp_path / "plain")
    assert path.endswith("plain.r2v")
    assert load_model(tmp_path / "plain").is_fitted
    assert load_model(path).is_fitted


def test_overwrite_protection(tmp_path: Path) -> None:
    model = fit("pca", messy_frame())
    save_model(model, tmp_path / "m")
    with pytest.raises(FileExistsError):
        save_model(model, tmp_path / "m")
    save_model(model, tmp_path / "m", overwrite=True)


def test_failed_save_leaves_the_old_file_and_no_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = fit("pca", messy_frame())
    path = Path(save_model(model, tmp_path / "m"))
    before = path.read_bytes()

    def boom(_obj: Any) -> bytes:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ser, "_dump_sklearn", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        save_model(model, tmp_path / "m", overwrite=True)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["m.r2v"]


def test_inspect_model_reads_the_manifest_only(tmp_path: Path) -> None:
    path = save_model(fit("pca", messy_frame()), tmp_path / "m")

    info = inspect_model(path)

    assert info["format"] == "row2vec-model"
    assert info["format_version"] == ser.FORMAT_VERSION
    assert info["row2vec_version"] == row2vec.__version__
    assert info["mode"] == "pca"
    assert info["libraries"]["scikit-learn"]
    assert "preprocessor.skops" in info["members"]


# --------------------------------------------------------------------------- #
# Loading never runs code, and refuses what it does not recognise
# --------------------------------------------------------------------------- #


def test_the_old_script_format_is_refused_and_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    script = tmp_path / "model.py"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('pwned')\n")

    with pytest.raises(ModelFormatError, match="no longer supported"):
        load_model(script)

    assert not marker.exists()


def test_a_type_off_the_allow_list_is_refused(tmp_path: Path) -> None:
    path = Path(save_model(fit("pca", messy_frame()), tmp_path / "m"))
    rewrite(path, replace={"preprocessor.skops": bytes(sio.dumps(Gadget()))})

    with pytest.raises(ModelFormatError, match="Gadget"):
        load_model(path)


def test_the_allow_list_is_exact_not_a_prefix(tmp_path: Path) -> None:
    """A row2vec class that is not listed is refused as well."""
    from row2vec.config import NeuralConfig

    assert "row2vec.config.NeuralConfig" not in ser.TRUSTED_TYPES
    path = Path(save_model(fit("pca", messy_frame()), tmp_path / "m"))
    rewrite(path, replace={"scaler.skops": bytes(sio.dumps(NeuralConfig()))})

    with pytest.raises(ModelFormatError, match="NeuralConfig"):
        load_model(path)


def test_a_modified_member_fails_its_checksum(tmp_path: Path) -> None:
    path = Path(save_model(fit("pca", messy_frame()), tmp_path / "m"))
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["state.json"] = members["state.json"].replace(b"2", b"3", 1)
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in members.items():
            archive.writestr(name, blob)

    with pytest.raises(ModelFormatError, match="checksum"):
        load_model(path)


@pytest.mark.parametrize("name", ["../escape.txt", "extra.py", "encoder.pkl"])
def test_unexpected_members_are_refused(tmp_path: Path, name: str) -> None:
    path = Path(save_model(fit("pca", messy_frame()), tmp_path / "m"))
    rewrite(path, replace={name: b"x"})

    with pytest.raises(ModelFormatError, match="unexpected members"):
        load_model(path)


def test_a_missing_member_is_refused(tmp_path: Path) -> None:
    path = Path(save_model(fit("pca", messy_frame()), tmp_path / "m"))
    rewrite(path, drop=("scaler.skops",))
    with pytest.raises(ModelFormatError, match="missing required"):
        load_model(path)


def test_a_newer_format_version_is_refused_with_advice(tmp_path: Path) -> None:
    path = Path(save_model(fit("pca", messy_frame()), tmp_path / "m"))
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["format_version"] = 99
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in members.items():
            archive.writestr(name, blob)

    with pytest.raises(ModelFormatError, match="Upgrade row2vec"):
        load_model(path)


def test_a_file_that_is_not_a_model_is_refused(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.r2v"
    bogus.write_bytes(b"not a zip")
    with pytest.raises(ModelFormatError, match="not a row2vec model"):
        load_model(bogus)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.r2v")


def test_a_callable_umap_metric_cannot_be_saved() -> None:
    fake = SimpleNamespace(metric=lambda _a, _b: 0.0, output_metric="euclidean")
    with pytest.raises(ModelFormatError, match="callable metric"):
        ser._strip_umap(fake)


@pytest.mark.neural
def test_an_encoder_with_a_lambda_layer_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    import keras  # type: ignore[import-untyped]

    inputs = keras.Input(shape=(3,))
    outputs = keras.layers.Lambda(lambda t: t * 2)(inputs)
    path = tmp_path / "evil.keras"
    keras.Model(inputs, outputs).save(path)

    with pytest.raises(ModelFormatError, match="layers row2vec does not write"):
        ser._load_keras(path.read_bytes())


@pytest.mark.neural
def test_neural_models_do_not_save_the_training_network(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    path = save_model(fit("unsupervised", messy_frame(), max_epochs=2), tmp_path / "m")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    assert "encoder.keras" in names
    assert "projector.skops" not in names
