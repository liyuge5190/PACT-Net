import numpy as np

from pactnet.camera import Camera, look_at
from pactnet.heatmap import heatmap_moments, information_matrix
from pactnet.triangulation import triangulate_robust


def _cameras():
    K = np.array([[1500.0, 0, 1024], [0, 1500.0, 768], [0, 0, 1]])
    pos = [(-5.6, 4.2, 1.35), (-2.8, 4.6, 1.7), (0.4, 4.2, 1.25), (-5.6, -4.2, 1.4), (-2.8, -4.6, 1.75), (0.4, -4.2, 1.3)]
    tgt = [(-4.5, 0, 1), (-2, 0, 1), (0.5, 0, 1)] * 2
    cams = []
    for p, t in zip(pos, tgt):
        R = look_at(p, t)
        cams.append(Camera(K, R, -R @ np.array(p), [-0.05, 0.01, 0.001, -0.001, 0.0]))
    return cams


def test_projection_jacobian():
    cam = _cameras()[0]
    X = np.array([[-2.0, 0.1, 1.0]])
    _, Jac = cam.project_with_jacobian(X)
    h = 1e-6
    num = np.stack([(cam.project(X + h * e) - cam.project(X - h * e)) / (2 * h) for e in np.eye(3)], -1)
    assert np.abs(num - Jac).max() < 1e-4


def test_undistortion_inverts_projection():
    cam = _cameras()[1]
    X = np.array([[-1.0, 0.2, 0.8], [-3.0, -0.3, 1.6]])
    xy = cam.undistort_normalized(cam.project(X))
    P = cam.to_camera(X)
    assert np.allclose(xy, P[:, :2] / P[:, 2:], atol=1e-9)


def test_heatmap_moments_are_anisotropic():
    ys, xs = np.meshgrid(np.arange(64), np.arange(48), indexing="ij")
    g = np.exp(-((xs - 20.3) ** 2 / (2 * 2.0 ** 2) + (ys - 30.7) ** 2 / (2 * 4.0 ** 2)))
    mu, cov, vis = heatmap_moments(g, kappa=0.05)
    assert np.allclose(mu, [20.3, 30.7], atol=0.1)
    assert cov[1, 1] > 2.5 * cov[0, 0]
    assert 0.0 < vis < 1.0


def test_robust_triangulation_with_outliers():
    rng = np.random.default_rng(0)
    cams = _cameras()
    N = 300
    X = np.c_[rng.uniform(-4, 0.5, N), rng.uniform(-0.5, 0.5, N), rng.uniform(0, 2, N)]
    mu = np.stack([c.project(X) for c in cams]) + rng.normal(0, 1.0, (6, N, 2))
    mu[0, :30] += 60.0
    info = information_matrix(np.tile(np.eye(2), (6, N, 1, 1)), np.ones((6, N)))
    Xh, cov = triangulate_robust(cams, mu, info)
    err = np.linalg.norm(Xh - X, axis=1)
    assert err.mean() < 0.005
    assert err[:30].mean() < 0.008
    assert np.all(np.linalg.eigvalsh(cov) > 0)
