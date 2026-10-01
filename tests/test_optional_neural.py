"""`pip install row2vec` must work without TensorFlow.

Each test runs in a fresh interpreter in which ``import tensorflow`` is made to
fail, so it checks the same thing whether or not TensorFlow is installed on the
machine running the suite.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_BLOCK_TENSORFLOW = (
    "import sys; sys.modules['tensorflow'] = None  # makes `import tensorflow` raise\n"
)


def run_without_tensorflow(body: str) -> subprocess.CompletedProcess[str]:
    code = _BLOCK_TENSORFLOW + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_import_does_not_need_tensorflow() -> None:
    result = run_without_tensorflow("import row2vec; print('ok')")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("mode", ["pca", "umap", "tsne"])
def test_classical_modes_work_without_tensorflow(mode: str) -> None:
    result = run_without_tensorflow(
        f"""
        import row2vec
        df = row2vec.generate_synthetic_data(200)
        out = row2vec.learn_embedding(df, mode={mode!r}, embedding_dim=2)
        assert out.shape == (200, 2), out.shape
        assert not any(m.startswith('tensorflow') and sys.modules[m] for m in sys.modules)
        print('ok')
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("mode", ["unsupervised", "contrastive"])
def test_neural_modes_say_how_to_get_tensorflow(mode: str) -> None:
    result = run_without_tensorflow(
        f"""
        import row2vec
        from row2vec._backend import NeuralBackendMissing
        df = row2vec.generate_synthetic_data(60)
        try:
            row2vec.learn_embedding(df, mode={mode!r}, embedding_dim=2, reference_column='Country')
        except NeuralBackendMissing as exc:
            assert 'row2vec[neural]' in str(exc), exc
            print('ok')
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_high_cardinality_without_a_target_explains_itself() -> None:
    """Entity embeddings are the one preprocessing path that needs TensorFlow."""
    result = run_without_tensorflow(
        """
        import pandas as pd
        import row2vec
        from row2vec._backend import NeuralBackendMissing
        df = pd.DataFrame({
            'id': [f'id_{i % 1500:04d}' for i in range(3000)],
            'x': range(3000),
        })
        try:
            row2vec.learn_embedding(df, mode='pca', embedding_dim=2)
        except NeuralBackendMissing as exc:
            assert 'categorical_encoding_strategy' in str(exc), exc
            print('ok')
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_high_cardinality_with_a_target_falls_back_to_target_encoding() -> None:
    result = run_without_tensorflow(
        """
        import warnings
        import numpy as np
        import pandas as pd
        import row2vec
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            'id': [f'id_{i % 1500:04d}' for i in range(3000)],
            'x': rng.normal(size=3000),
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            from row2vec.categorical_encoding import CategoricalAnalyzer, CategoricalEncodingConfig
            analysis = CategoricalAnalyzer(CategoricalEncodingConfig()).analyze_column(
                df['id'], pd.Series(rng.normal(size=3000))
            )
        assert analysis['recommended_strategy'] == 'target', analysis
        assert any('row2vec[neural]' in str(w.message) for w in caught)
        print('ok')
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
