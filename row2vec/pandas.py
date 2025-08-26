"""
Pandas integration for Row2Vec embeddings.

This module provides a pandas accessor that allows direct embedding
generation from DataFrames using the `.row2vec` accessor.
"""

from typing import Any

import pandas as pd

from .api import learn_embedding_v2
from .config import EmbeddingConfig, create_config_for_mode


@pd.api.extensions.register_dataframe_accessor("row2vec")
class Row2VecAccessor:
    """
    Pandas DataFrame accessor for Row2Vec embeddings.

    This accessor provides convenient methods to generate embeddings
    directly from DataFrames using the .row2vec namespace.

    Examples
    --------
    >>> import pandas as pd
    >>> import row2vec  # This registers the accessor
    >>>
    >>> # Basic usage
    >>> df = pd.DataFrame(...)
    >>> embeddings = df.row2vec.embed(dim=10)
    >>>
    >>> # With specific mode
    >>> embeddings = df.row2vec.embed(dim=15, mode="contrastive")
    >>>
    >>> # With full configuration
    >>> embeddings = df.row2vec.unsupervised(dim=10, max_epochs=100)
    >>> embeddings = df.row2vec.contrastive(dim=20, loss_type="triplet")
    >>> embeddings = df.row2vec.classical(method="pca", dim=5)
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj

    def embed(
        self,
        dim: int = 10,
        mode: str = "unsupervised",
        config: EmbeddingConfig | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generate embeddings for the DataFrame.

        Parameters
        ----------
        dim : int, default=10
            Dimensionality of the embedding space.
        mode : str, default="unsupervised"
            Embedding mode. Options: "unsupervised", "target", "pca", "tsne", "umap", "contrastive".
        config : EmbeddingConfig, optional
            Pre-configured EmbeddingConfig object. If provided, other parameters are ignored.
        **kwargs
            Additional parameters for embedding configuration.

        Returns
        -------
        embeddings : DataFrame
            DataFrame with embedded representations.

        Examples
        --------
        >>> embeddings = df.row2vec.embed(dim=10, mode="unsupervised")
        >>> embeddings = df.row2vec.embed(dim=15, mode="contrastive", loss_type="triplet")
        """
        if config is None:
            # Create config with specified parameters
            config = create_config_for_mode(mode)
            config.embedding_dim = dim

            # Apply any additional kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                elif "." in key:
                    # Handle nested parameters like neural.max_epochs
                    section, param = key.split(".", 1)
                    if hasattr(config, section):
                        section_config = getattr(config, section)
                        if hasattr(section_config, param):
                            setattr(section_config, param, value)

        return learn_embedding_v2(self._obj, config, **kwargs)

    def unsupervised(
        self,
        dim: int = 10,
        max_epochs: int = 50,
        batch_size: int = 64,
        dropout_rate: float = 0.2,
        hidden_units: int = 128,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generate unsupervised embeddings using autoencoder.

        Parameters
        ----------
        dim : int, default=10
            Dimensionality of the embedding space.
        max_epochs : int, default=50
            Maximum number of training epochs.
        batch_size : int, default=64
            Training batch size.
        dropout_rate : float, default=0.2
            Dropout rate for regularization.
        hidden_units : int, default=128
            Number of hidden units in the autoencoder.
        **kwargs
            Additional parameters passed to the config.

        Returns
        -------
        embeddings : DataFrame
            DataFrame with unsupervised embeddings.

        Examples
        --------
        >>> embeddings = df.row2vec.unsupervised(dim=10, max_epochs=100)
        """
        config = create_config_for_mode("unsupervised")
        config.embedding_dim = dim
        config.neural.max_epochs = max_epochs
        config.neural.batch_size = batch_size
        config.neural.dropout_rate = dropout_rate
        config.neural.hidden_units = hidden_units

        return learn_embedding_v2(self._obj, config, **kwargs)

    def supervised(
        self,
        target_column: str,
        dim: int = 10,
        max_epochs: int = 50,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generate supervised embeddings using target column.

        Parameters
        ----------
        target_column : str
            Name of the target column for supervised learning.
        dim : int, default=10
            Dimensionality of the embedding space.
        max_epochs : int, default=50
            Maximum number of training epochs.
        batch_size : int, default=64
            Training batch size.
        **kwargs
            Additional parameters passed to the config.

        Returns
        -------
        embeddings : DataFrame
            DataFrame with supervised embeddings.

        Examples
        --------
        >>> embeddings = df.row2vec.supervised("category", dim=15)
        """
        config = create_config_for_mode("target")
        config.embedding_dim = dim
        config.reference_column = target_column
        config.neural.max_epochs = max_epochs
        config.neural.batch_size = batch_size

        return learn_embedding_v2(self._obj, config, **kwargs)

    def contrastive(
        self,
        dim: int = 10,
        loss_type: str = "triplet",
        auto_pairs: str = "cluster",
        margin: float = 1.0,
        negative_samples: int = 5,
        max_epochs: int = 100,
        batch_size: int = 32,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate contrastive embeddings using similarity learning.

        Parameters
        ----------
        dim : int, default=10
            Dimensionality of the embedding space.
        loss_type : str, default="triplet"
            Type of contrastive loss. Options: "triplet", "contrastive".
        auto_pairs : str, default="cluster"
            Strategy for automatic pair generation. Options: "cluster", "neighbors", "random".
        margin : float, default=1.0
            Margin for the contrastive loss.
        negative_samples : int, default=5
            Number of negative samples per positive pair.
        max_epochs : int, default=100
            Maximum number of training epochs.
        batch_size : int, default=32
            Training batch size.
        **kwargs
            Additional parameters passed to the config.

        Returns
        -------
        embeddings : DataFrame
            DataFrame with contrastive embeddings.

        Examples
        --------
        >>> embeddings = df.row2vec.contrastive(dim=20, loss_type="triplet")
        """
        config = create_config_for_mode("contrastive")
        config.embedding_dim = dim
        config.contrastive.loss_type = loss_type
        config.contrastive.auto_pairs = auto_pairs
        config.contrastive.margin = margin
        config.contrastive.negative_samples = negative_samples
        config.neural.max_epochs = max_epochs
        config.neural.batch_size = batch_size

        return learn_embedding_v2(self._obj, config, **kwargs)

    def classical(
        self,
        method: str = "pca",
        dim: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate embeddings using classical ML methods.

        Parameters
        ----------
        method : str, default="pca"
            Classical method to use. Options: "pca", "tsne", "umap".
        dim : int, default=10
            Dimensionality of the embedding space.
        **kwargs
            Additional parameters for the specific method:
            - For t-SNE: perplexity, n_iter
            - For UMAP: n_neighbors, min_dist

        Returns
        -------
        embeddings : DataFrame
            DataFrame with classical embeddings.

        Examples
        --------
        >>> # PCA embeddings
        >>> embeddings = df.row2vec.classical("pca", dim=5)
        >>>
        >>> # t-SNE embeddings
        >>> embeddings = df.row2vec.classical("tsne", dim=2, perplexity=30)
        >>>
        >>> # UMAP embeddings
        >>> embeddings = df.row2vec.classical("umap", dim=3, n_neighbors=15)
        """
        config = create_config_for_mode(method)
        config.embedding_dim = dim

        # Apply method-specific parameters
        if method == "tsne":
            if "perplexity" in kwargs:
                config.classical.perplexity = kwargs.pop("perplexity")
            if "n_iter" in kwargs:
                config.classical.n_iter = kwargs.pop("n_iter")
        elif method == "umap":
            if "n_neighbors" in kwargs:
                config.classical.n_neighbors = kwargs.pop("n_neighbors")
            if "min_dist" in kwargs:
                config.classical.min_dist = kwargs.pop("min_dist")

        return learn_embedding_v2(self._obj, config, **kwargs)

    def pca(self, dim: int = 10, **kwargs) -> pd.DataFrame:
        """Generate PCA embeddings."""
        return self.classical("pca", dim=dim, **kwargs)

    def tsne(
        self,
        dim: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate t-SNE embeddings."""
        return self.classical(
            "tsne",
            dim=dim,
            perplexity=perplexity,
            n_iter=n_iter,
            **kwargs,
        )

    def umap(
        self,
        dim: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate UMAP embeddings."""
        return self.classical(
            "umap",
            dim=dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            **kwargs,
        )

    def compare_methods(
        self,
        dim: int = 5,
        methods: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Compare multiple embedding methods on the same data.

        Parameters
        ----------
        dim : int, default=5
            Dimensionality for all methods.
        methods : list, optional
            List of methods to compare. If None, uses ["unsupervised", "pca", "tsne", "umap"].

        Returns
        -------
        results : dict
            Dictionary mapping method names to their embedding DataFrames.

        Examples
        --------
        >>> results = df.row2vec.compare_methods(dim=3)
        >>> print("PCA variance:", results["pca"].var().mean())
        >>> print("t-SNE variance:", results["tsne"].var().mean())
        """
        if methods is None:
            methods = ["unsupervised", "pca", "tsne", "umap"]

        results = {}
        for method in methods:
            try:
                if method == "unsupervised":
                    results[method] = self.unsupervised(dim=dim)
                elif method in ["pca", "tsne", "umap"]:
                    results[method] = self.classical(method=method, dim=dim)
                elif method == "contrastive":
                    results[method] = self.contrastive(dim=dim)
                else:
                    # Try to use it as a mode directly
                    results[method] = self.embed(dim=dim, mode=method)
            except Exception:
                continue

        return results
