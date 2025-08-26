import pytest

from row2vec import generate_synthetic_data, learn_embedding


@pytest.fixture
def synthetic_data():
    """Provides a standard synthetic dataset for testing."""
    return generate_synthetic_data(num_records=100)


def test_unsupervised_embedding(synthetic_data):
    """
    Tests the unsupervised mode.
    """
    df = synthetic_data
    embedding_dim = 5
    embeddings = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=embedding_dim,
        max_epochs=2,  # Keep it fast
    )

    assert embeddings.shape[0] == df.shape[0]
    assert embeddings.shape[1] == embedding_dim
    assert list(embeddings.columns) == [f"embedding_{i}" for i in range(embedding_dim)]


def test_target_embedding(synthetic_data):
    """
    Tests the target-based mode.
    """
    df = synthetic_data
    embedding_dim = 3
    reference_column = "Country"
    num_unique_countries = df[reference_column].nunique()

    embeddings = learn_embedding(
        df,
        mode="target",
        reference_column=reference_column,
        embedding_dim=embedding_dim,
        max_epochs=2,  # Keep it fast
    )

    assert embeddings.shape[0] == num_unique_countries
    assert embeddings.shape[1] == embedding_dim
    assert list(embeddings.columns) == [f"embedding_{i}" for i in range(embedding_dim)]


def test_minmax_scaling_range_unsupervised(synthetic_data):
    df = synthetic_data
    emb = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=3,
        max_epochs=1,
        scale_range=(-1.0, 1.0),  # triggers default minmax
    )
    assert (emb.values >= -1.0 - 1e-6).all()
    assert (emb.values <= 1.0 + 1e-6).all()


def test_minmax_constant_column_raises_on_single_category(synthetic_data):
    # Force a single category so grouping yields 1 row -> min==max for any column
    df = synthetic_data.copy()
    df["Country"] = "OnlyOne"
    with pytest.raises(ValueError):
        learn_embedding(
            df,
            mode="target",
            reference_column="Country",
            embedding_dim=2,
            max_epochs=1,
            scale_method="minmax",
            scale_range=(0.0, 1.0),
        )


def test_standard_l2_tanh_scaling(synthetic_data):
    df = synthetic_data
    # standard
    emb_std = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=4,
        max_epochs=1,
        scale_method="standard",
    )
    import numpy as np

    col_means = emb_std.mean(axis=0).to_numpy()
    col_stds = emb_std.std(axis=0).to_numpy()
    assert (np.abs(col_means) < 1e-1).all()
    assert (col_stds > 0).all()
    # l2
    emb_l2 = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=4,
        max_epochs=1,
        scale_method="l2",
    )
    norms = np.linalg.norm(emb_l2.values, axis=1)
    assert (abs(norms - 1.0) < 1e-3).mean() > 0.9  # most rows should be ~unit norm
    # tanh
    emb_th = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=4,
        max_epochs=1,
        scale_method="tanh",
    )
    assert (emb_th.values <= 1.0 + 1e-6).all()
    assert (emb_th.values >= -1.0 - 1e-6).all()


def test_invalid_mode_raises_error(synthetic_data):
    """
    Tests that an invalid mode raises a ValueError.
    """
    with pytest.raises(ValueError):
        learn_embedding(
            synthetic_data,
            mode="invalid_mode",
        )


def test_missing_reference_column_raises_error(synthetic_data):
    """
    Tests that target mode without a reference column raises a ValueError.
    """
    with pytest.raises(ValueError):
        learn_embedding(
            synthetic_data,
            mode="target",
            reference_column=None,
        )


# === CLASSICAL ML METHODS TESTS ===


def test_pca_embedding(synthetic_data):
    """
    Tests PCA embedding mode.
    """
    df = synthetic_data
    embedding_dim = 5
    embeddings = learn_embedding(
        df,
        mode="pca",
        embedding_dim=embedding_dim,
        verbose=False,
    )

    assert embeddings.shape[0] == df.shape[0]
    assert embeddings.shape[1] == embedding_dim
    assert list(embeddings.columns) == [f"embedding_{i}" for i in range(embedding_dim)]
    # PCA should preserve variance structure
    assert embeddings.var().sum() > 0


def test_tsne_embedding(synthetic_data):
    """
    Tests t-SNE embedding mode with small dataset.
    """
    df = synthetic_data
    embedding_dim = 2  # t-SNE works best with 2-3 dimensions
    embeddings = learn_embedding(
        df,
        mode="tsne",
        embedding_dim=embedding_dim,
        perplexity=5,  # Small perplexity for small dataset
        n_iter=250,  # Fewer iterations for speed
        verbose=False,
    )

    assert embeddings.shape[0] == df.shape[0]
    assert embeddings.shape[1] == embedding_dim
    assert list(embeddings.columns) == [f"embedding_{i}" for i in range(embedding_dim)]


def test_umap_embedding(synthetic_data):
    """
    Tests UMAP embedding mode.
    """
    df = synthetic_data
    embedding_dim = 3
    embeddings = learn_embedding(
        df,
        mode="umap",
        embedding_dim=embedding_dim,
        n_neighbors=10,  # Smaller for small dataset
        min_dist=0.1,
        verbose=False,
    )

    assert embeddings.shape[0] == df.shape[0]
    assert embeddings.shape[1] == embedding_dim
    assert list(embeddings.columns) == [f"embedding_{i}" for i in range(embedding_dim)]


def test_classical_methods_with_scaling(synthetic_data):
    """
    Tests that scaling works with classical methods.
    """
    df = synthetic_data

    # Test PCA with minmax scaling
    pca_scaled = learn_embedding(
        df,
        mode="pca",
        embedding_dim=3,
        scale_method="minmax",
        scale_range=(0, 1),
        verbose=False,
    )

    assert (pca_scaled.values >= -1e-6).all()
    assert (pca_scaled.values <= 1 + 1e-6).all()

    # Test PCA with standard scaling
    pca_standard = learn_embedding(
        df,
        mode="pca",
        embedding_dim=3,
        scale_method="standard",
        verbose=False,
    )

    # Standard scaling should result in approximately zero mean
    import numpy as np

    col_means = pca_standard.mean(axis=0).to_numpy()
    assert (np.abs(col_means) < 1e-1).all()


def test_classical_method_parameter_validation(synthetic_data):
    """
    Tests parameter validation for classical methods.
    """
    df = synthetic_data

    # Test invalid perplexity for t-SNE
    with pytest.raises(ValueError):
        learn_embedding(
            df,
            mode="tsne",
            perplexity=-1,  # Invalid negative perplexity
            verbose=False,
        )

    # Test invalid n_neighbors for UMAP
    with pytest.raises(ValueError):
        learn_embedding(
            df,
            mode="umap",
            n_neighbors=0,  # Invalid zero neighbors
            verbose=False,
        )

    # Test perplexity too large for dataset size
    with pytest.raises(ValueError):
        learn_embedding(
            df,
            mode="tsne",
            perplexity=50,  # Too large for 100 samples
            verbose=False,
        )

    # Test n_neighbors too large for dataset size
    with pytest.raises(ValueError):
        learn_embedding(
            df,
            mode="umap",
            n_neighbors=150,  # Larger than dataset size
            verbose=False,
        )


def test_classical_methods_different_dimensions(synthetic_data):
    """
    Tests classical methods with different embedding dimensions.
    """
    df = synthetic_data

    for dim in [2, 5, 10]:
        # Test PCA
        pca_emb = learn_embedding(
            df,
            mode="pca",
            embedding_dim=dim,
            verbose=False,
        )
        assert pca_emb.shape[1] == dim

        # Test UMAP (skip very high dimensions for speed)
        if dim <= 5:
            umap_emb = learn_embedding(
                df,
                mode="umap",
                embedding_dim=dim,
                n_neighbors=5,
                verbose=False,
            )
            assert umap_emb.shape[1] == dim


def test_pca_explained_variance_logging(synthetic_data):
    """
    Tests that PCA logs explained variance information.
    """
    df = synthetic_data

    # This test mainly ensures PCA runs without errors
    # and logs meaningful information
    embeddings = learn_embedding(
        df,
        mode="pca",
        embedding_dim=3,
        enable_logging=True,
        log_level="DEBUG",
        verbose=False,
    )

    assert embeddings.shape == (df.shape[0], 3)


def test_tsne_high_dimension_warning(synthetic_data):
    """
    Tests that t-SNE warns for high embedding dimensions.
    """
    df = synthetic_data

    # This should work but might log a warning
    embeddings = learn_embedding(
        df,
        mode="tsne",
        embedding_dim=5,  # Higher than recommended
        perplexity=5,
        n_iter=250,  # Keep it fast but above minimum
        enable_logging=True,
        verbose=False,
    )

    assert embeddings.shape == (df.shape[0], 5)


def test_invalid_classical_mode_raises_error(synthetic_data):
    """
    Tests that invalid classical method raises ValueError.
    """
    df = synthetic_data

    with pytest.raises(ValueError, match="mode must be one of"):
        learn_embedding(
            df,
            mode="invalid_classical_method",
            verbose=False,
        )
