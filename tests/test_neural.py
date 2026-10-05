"""Tests for Neural Network framework, vectorized inference, and topology evolution."""
import pytest
import numpy as np

from src.neural.network import NetSpec, init_params, forward, forward_one
from src.neural.neuroevolution import structural_mutation, topology_distance
from src.neural.backend import get_backend


def test_netspec_geometry():
    spec = NetSpec(n_in=12, hidden=[8, 6], recurrent=False)
    assert spec.n_in == 12
    assert spec.n_out == 7
    assert spec.n_params > 0
    assert len(spec.offsets) == 3  # in->h1, h1->h2, h2->out


def test_batched_forward_shape():
    spec = NetSpec(n_in=10, hidden=[6, 4], recurrent=False)
    rng = np.random.default_rng(42)
    n_orgs = 50
    w = init_params(spec, rng, n_orgs)
    x = rng.normal(0, 1, (n_orgs, 10)).astype(np.float32)

    y, h_next = forward(spec, w, x, h_prev=None)
    assert y.shape == (n_orgs, spec.n_out)
    assert h_next.shape == (n_orgs, spec.state_size)
    assert np.all(np.isfinite(y))


def test_recurrent_forward():
    spec = NetSpec(n_in=8, hidden=[6], recurrent=True)
    rng = np.random.default_rng(42)
    n_orgs = 20
    w = init_params(spec, rng, n_orgs)
    x = rng.normal(0, 1, (n_orgs, 8)).astype(np.float32)
    h0 = np.zeros((n_orgs, spec.state_size), dtype=np.float32)

    y1, h1 = forward(spec, w, x, h_prev=h0)
    assert y1.shape == (n_orgs, spec.n_out)
    assert h1.shape == (n_orgs, spec.state_size)

    # Subsequent forward step retains and alters state
    y2, h2 = forward(spec, w, x, h_prev=h1)
    assert not np.array_equal(h1, h2)


def test_single_forward_consistency():
    spec = NetSpec(n_in=8, hidden=[6], recurrent=False)
    rng = np.random.default_rng(42)
    w = init_params(spec, rng, 1)[0]
    x = rng.normal(0, 1, 8).astype(np.float32)

    y_single, st, acts = forward_one(spec, w, x)
    w_batch = w.reshape(1, -1)
    x_batch = x.reshape(1, -1)
    y_batch, _ = forward(spec, w_batch, x_batch)

    np.testing.assert_allclose(y_single, y_batch[0], atol=1e-5)


def test_nan_sanitation():
    from src.core.config import Config
    from src.simulation.ecosystem import Ecosystem

    cfg = Config()
    cfg.population = 10
    eco = Ecosystem(config=cfg, seed=42)
    # Inject NaN into organism weights
    eco.org.w[0] = np.nan
    eco.org.w_eff[0] = np.nan
    eco.step()

    # NaN must be quarantined and output sanitized
    assert eco.nan_events >= 1
    assert np.all(np.isfinite(eco.org.last_out[:eco.org.n]))


def test_topology_mutation_and_distance():
    spec = NetSpec(n_in=6, hidden=[4], recurrent=False)
    rng = np.random.default_rng(42)
    masks = np.ones((2, spec.n_params), dtype=np.uint8)
    weights = init_params(spec, rng, 2)

    # Initially zero distance
    assert topology_distance(masks[0], masks[1], spec) == 0.0

    # Mutate topology of row 1
    rate = np.array([0.0, 0.5], dtype=np.float32)
    structural_mutation(weights, masks, rate, spec, rng)
    assert topology_distance(masks[0], masks[1], spec) >= 0.0
