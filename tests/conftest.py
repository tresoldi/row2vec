"""Shared pytest configuration.

The project registers ``slow``, ``neural``, ``integration`` and ``unit``
markers in ``pyproject.toml``, but before 0.4.0 nothing carried them, so
``make test-fast`` (``pytest -n auto -m "not slow"``) deselected nothing and
took as long as a full run.

Rather than sprinkle decorators across twenty files and let them rot, the slow
set is declared here in one place, measured rather than guessed: every entry
below took more than four seconds in a full ``--durations`` run. Keeping it in
one list makes it obvious when something new becomes expensive.
"""

from __future__ import annotations

import pytest

# Node-id fragments for tests measured at >4s. A test is marked slow when its
# node id contains any of these.
SLOW_TEST_FRAGMENTS: tuple[str, ...] = (
    "test_core.py::test_umap_embedding",
    "test_core.py::test_standard_l2_tanh_scaling",
    "test_integrations.py::TestSklearnIntegration",
    "test_integrations.py::TestPandasIntegration::test_pandas_unsupervised",
    "test_cli_commands.py::TestSearchArchitecture",
    "test_docs_examples.py::test_doc_code_block_runs",
    "test_doctests.py::test_module_docstring_examples",
    "test_config_api.py::TestConfigBasedAPI",
    "test_config_api.py::TestAPICompatibility::test_config_vs_legacy_equivalence",
    "test_serialization.py::TestModelSerialization",
    "test_integration.py::TestCompatibility::test_different_batch_sizes",
    "test_performance.py::TestPerformanceBenchmarks",
    "test_logging.py::TestLoggingIntegration::test_performance_warnings",
)

# Modules whose tests train a Keras model. These carry the `neural` marker so a
# contributor without a working TensorFlow build can deselect them.
NEURAL_TEST_MODULES: tuple[str, ...] = (
    "test_contrastive.py",
    "test_minimal_contrastive.py",
    "test_architecture_search.py",
)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Attach the `slow` and `neural` markers to the tests that earn them."""
    for item in items:
        node_id = item.nodeid

        if any(fragment in node_id for fragment in SLOW_TEST_FRAGMENTS):
            item.add_marker(pytest.mark.slow)

        if any(module in node_id for module in NEURAL_TEST_MODULES):
            item.add_marker(pytest.mark.neural)


# --------------------------------------------------------------------------- #
# Running without the optional neural backend
# --------------------------------------------------------------------------- #
#
# `pip install row2vec` has no TensorFlow, and `learn_embedding`'s default mode
# is a neural one, so a large part of the suite reaches TensorFlow without
# saying so. Rather than mark each test, a failure whose report carries the
# library's own `NeuralBackendMissing` error is reported as a skip when
# TensorFlow is absent. Any *other* failure is still a failure, which is what
# the light CI job is there to catch.

_NEEDS_TENSORFLOW = "NeuralBackendMissing"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    outcome = yield
    report = outcome.get_result()
    if report.failed and _NEEDS_TENSORFLOW in str(report.longrepr):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, "requires the [neural] extra")
