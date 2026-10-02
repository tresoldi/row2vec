"""Text columns: opt-in detection, the TF-IDF default, the hook, and persistence."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import row2vec
from row2vec.config import EmbeddingConfig, PreprocessingConfig
from row2vec.pipeline_builder import PipelineBuilder
from row2vec.serialization import ModelFormatError, inspect_model, load_model, save_model
from row2vec.text_encoding import TextEncoder, detached_hooks

DOCS = [
    "red apple pie",
    "green apple tart",
    "blue sky today",
    "clear blue sky",
    "apple pie recipe",
    "sunny sky weather",
]


def _frame(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(float(n)), "note": [DOCS[i % len(DOCS)] for i in range(n)]})


def _config(**preprocessing: Any) -> EmbeddingConfig:
    cfg = EmbeddingConfig(mode="pca", embedding_dim=3)
    cfg.preprocessing = PreprocessingConfig(text_columns=["note"], text_dim=4, **preprocessing)
    return cfg


def _hook(texts: list[str]) -> np.ndarray:
    return np.array([[len(t), t.count("a"), t.count("e")] for t in texts], dtype=float)


def _fit(cfg: EmbeddingConfig, df: pd.DataFrame | None = None) -> Any:
    _, model = row2vec.learn_embedding_with_model(
        _frame() if df is None else df,
        enable_logging=False,
        config=cfg,
        mode="pca",
        embedding_dim=3,
    )
    return model


# -- the default encoder ---------------------------------------------------- #


def test_default_encoder_shape_names_and_missing_values() -> None:
    df = _frame()
    df.loc[3, "note"] = None
    enc = TextEncoder(n_components=4).fit(df[["note"]])
    out = enc.transform(df[["note"]])
    assert out.shape == (60, 4)
    assert np.isfinite(out).all()
    assert list(enc.get_feature_names_out()) == [f"note_text_{i}" for i in range(4)]
    assert not enc.uses_hook_


def test_width_is_clamped_to_the_vocabulary() -> None:
    enc = TextEncoder(n_components=50).fit(pd.DataFrame({"t": ["a b", "b c", "c a"] * 4}))
    assert enc.transform(pd.DataFrame({"t": ["a b"]})).shape[1] <= 3


def test_similar_texts_land_closer_than_different_ones() -> None:
    enc = TextEncoder(n_components=3).fit(_frame()[["note"]])
    a, b, c = enc.transform(pd.DataFrame({"note": ["apple pie", "apple tart", "blue sky"]}))
    assert np.linalg.norm(a - b) < np.linalg.norm(a - c)


def test_text_with_no_words_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="no usable words"):
        TextEncoder().fit(pd.DataFrame({"t": ["", "", ""]}))


# -- the hook ---------------------------------------------------------------- #


def test_hook_is_used_batched_and_validated() -> None:
    calls: list[int] = []

    def hook(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return _hook(texts)

    enc = TextEncoder(encoder=hook, batch_size=25).fit(_frame()[["note"]])
    assert enc.uses_hook_
    assert enc.hook_dims_ == [3]
    assert calls == [25, 25, 10]
    assert enc.transform(_frame(7)[["note"]]).shape == (7, 3)
    assert list(enc.get_feature_names_out()) == ["note_text_0", "note_text_1", "note_text_2"]


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (lambda t: np.zeros(len(t)), "shape"),
        (lambda t: np.zeros((len(t) + 1, 2)), "shape"),
        (lambda t: np.full((len(t), 2), np.nan), "non-finite"),
    ],
)
def test_bad_hook_output_is_rejected(bad: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TextEncoder(encoder=bad).fit(_frame()[["note"]])


def test_hook_width_must_not_change_after_fit() -> None:
    state = {"dim": 3}
    enc = TextEncoder(encoder=lambda t: np.zeros((len(t), state["dim"]))).fit(_frame()[["note"]])
    state["dim"] = 5
    with pytest.raises(ValueError, match="fitted with 3"):
        enc.transform(_frame(4)[["note"]])


def test_hook_is_shared_not_copied_by_clone() -> None:
    class Heavy:
        def __call__(self, texts: list[str]) -> np.ndarray:
            return _hook(texts)

        def __deepcopy__(self, memo: Any) -> Any:
            raise AssertionError("the hook was deep-copied")

    cfg = _config(text_encoder=Heavy())
    pre, _ = PipelineBuilder(cfg).build_preprocessing_pipeline(_frame())
    pre.fit(_frame())  # ColumnTransformer clones its transformers here
    assert pre.transform(_frame()).shape[0] == 60


# -- configuration and routing ---------------------------------------------- #


def test_config_validation_and_yaml_round_trip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="text_dim"):
        PreprocessingConfig(text_dim=0)
    with pytest.raises(ValueError, match="list of column names"):
        PreprocessingConfig(text_columns="note")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="callable"):
        PreprocessingConfig(text_encoder="nope")  # type: ignore[arg-type]
    cfg = _config(text_encoder=_hook)
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert "text_encoder" not in path.read_text()
    restored = EmbeddingConfig.from_yaml(path)
    assert restored.preprocessing.text_columns == ["note"]
    assert restored.preprocessing.text_dim == 4
    assert restored.preprocessing.text_encoder is None


def test_text_columns_are_opt_in() -> None:
    pre, _ = PipelineBuilder(EmbeddingConfig()).build_preprocessing_pipeline(_frame())
    assert {name for name, *_ in pre.transformers} == {"numeric", "categorical"}


def test_text_column_leaves_the_categorical_branch() -> None:
    pre, analysis = PipelineBuilder(_config()).build_preprocessing_pipeline(_frame())
    assert {name for name, *_ in pre.transformers} == {"numeric", "text"}
    assert analysis["text_columns"] == 1


def test_text_only_frame_works() -> None:
    cfg = _config()
    out = row2vec.learn_embedding(
        _frame()[["note"]], mode="pca", embedding_dim=2, enable_logging=False, config=cfg
    )
    assert out.shape == (60, 2)


def test_missing_and_non_string_text_columns_are_errors() -> None:
    with pytest.raises(ValueError, match="not found"):
        PipelineBuilder(_config()).build_preprocessing_pipeline(_frame().drop(columns="note"))
    cfg = _config()
    cfg.preprocessing.text_columns = ["x"]
    with pytest.raises(ValueError, match="must be string columns"):
        PipelineBuilder(cfg).build_preprocessing_pipeline(_frame())


def test_target_cannot_be_a_text_column() -> None:
    df = _frame()
    with pytest.raises(ValueError, match="cannot also be a text column"):
        PipelineBuilder(_config()).build_preprocessing_pipeline(df, df["note"], mode="target")


# -- persistence ------------------------------------------------------------- #


def test_default_encoder_round_trips_exactly(tmp_path: Path) -> None:
    model = _fit(_config())
    loaded = load_model(save_model(model, tmp_path / "m"))
    new = _frame(15).assign(note=["apple pie sky"] * 15)
    assert np.allclose(
        model.predict(new, validate_schema=False).to_numpy(),
        loaded.predict(new, validate_schema=False).to_numpy(),
    )
    assert inspect_model(tmp_path / "m")["text_encoder_required"] is False


def test_custom_hook_is_not_saved_and_must_be_supplied(tmp_path: Path) -> None:
    model = _fit(_config(text_encoder=_hook))
    path = save_model(model, tmp_path / "h")
    assert inspect_model(path)["text_encoder_required"] is True
    with pytest.raises(ModelFormatError, match=r"load_model\(path, text_encoder="):
        load_model(path)
    loaded = load_model(path, text_encoder=_hook)
    expected = model.predict(_frame(), validate_schema=False).to_numpy()
    assert np.allclose(loaded.predict(_frame(), validate_schema=False).to_numpy(), expected)


def test_saving_leaves_the_live_model_usable(tmp_path: Path) -> None:
    model = _fit(_config(text_encoder=_hook))
    save_model(model, tmp_path / "h")
    assert model.predict(_frame(), validate_schema=False).shape == (60, 3)


def test_a_lambda_hook_can_be_saved_because_it_is_never_written(tmp_path: Path) -> None:
    model = _fit(_config(text_encoder=lambda t: _hook(t)))
    path = save_model(model, tmp_path / "l")
    assert load_model(path, text_encoder=_hook) is not None


def test_a_hook_with_the_wrong_width_is_caught_on_use(tmp_path: Path) -> None:
    path = save_model(_fit(_config(text_encoder=_hook)), tmp_path / "w")
    loaded = load_model(path, text_encoder=lambda t: np.zeros((len(t), 7)))
    with pytest.raises(ValueError, match="fitted with 3"):
        loaded.predict(_frame(), validate_schema=False)


def test_text_encoder_for_a_model_without_one_is_an_error(tmp_path: Path) -> None:
    path = save_model(_fit(_config()), tmp_path / "d")
    with pytest.raises(ValueError, match="does not use a custom text encoder"):
        load_model(path, text_encoder=_hook)


def test_detached_hooks_restores_on_error() -> None:
    cfg = _config(text_encoder=_hook)
    pre, _ = PipelineBuilder(cfg).build_preprocessing_pipeline(_frame())
    pre.fit(_frame())
    seen: list[TextEncoder] = []

    def use_and_fail() -> None:
        with detached_hooks(pre) as hooked:
            seen.extend(hooked)
            assert all(e.encoder is None for e in hooked)
            raise RuntimeError

    with pytest.raises(RuntimeError):
        use_and_fail()
    assert seen
    assert all(e.encoder is not None for e in seen)
    assert copy.copy(seen[0].encoder) is seen[0].encoder
