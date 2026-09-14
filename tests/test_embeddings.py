import numpy as np

from pipeline_sentinel.embeddings import DinoV2Embedder


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class FakeTorch:
    cuda = FakeCuda()


class FakeModel:
    def __init__(self) -> None:
        self.target = None

    def eval(self) -> None:
        pass

    def to(self, target: str) -> None:
        self.target = target


def test_dinov2_preparation_letterboxes_without_stretching() -> None:
    embedder = DinoV2Embedder(image_size=224, model=object())
    crop = np.full((50, 100, 3), 255, dtype=np.uint8)

    prepared = embedder._prepare(crop)

    assert prepared.shape == (3, 224, 224)
    # The 2:1 crop becomes a centered 224x112 band. Mean-colored padding normalizes to zero.
    assert np.allclose(prepared[:, :56], 0.0, atol=0.02)
    assert np.all(prepared[:, 56:168] > 1.0)
    assert np.allclose(prepared[:, 168:], 0.0, atol=0.02)


def test_dinov2_preparation_accepts_grayscale_crops() -> None:
    embedder = DinoV2Embedder(image_size=64, model=object())

    prepared = embedder._prepare(np.full((20, 20), 128, dtype=np.uint8))

    assert prepared.shape == (3, 64, 64)
    assert np.isfinite(prepared).all()


def test_integer_device_selects_cuda_index() -> None:
    model = FakeModel()
    embedder = DinoV2Embedder(device=1, model=model)
    embedder._torch = FakeTorch()

    embedder._ensure_runtime()

    assert embedder.device == "cuda:1"
    assert model.target == "cuda:1"
