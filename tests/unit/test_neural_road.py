from contextlib import nullcontext

import numpy as np
import pytest
import torch

from src.roads.neural_road import Mask2FormerRoadHighlighter


class FakeProcessor:
    def __call__(self, *, images: object, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 4, 4))}

    def post_process_semantic_segmentation(
        self, outputs: object, *, target_sizes: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        assert target_sizes == [(4, 4)]
        return [torch.tensor([[3, 0, 0, 3], [0, 0, 3, 0], [0, 0, 0, 0], [3, 0, 0, 0]])]


class FakeModel:
    def __call__(self, **inputs: torch.Tensor) -> object:
        assert "pixel_values" in inputs
        return object()


def test_mask2former_highlighter_builds_orange_road_overlay() -> None:
    highlighter = Mask2FormerRoadHighlighter.__new__(Mask2FormerRoadHighlighter)
    highlighter.road_class_id = 3
    highlighter.overlay_color = np.array([242, 107, 56], dtype=np.uint8)
    highlighter.opacity = 0.40
    highlighter._torch = type("TorchStub", (), {"inference_mode": staticmethod(nullcontext)})()
    highlighter.device = "cpu"
    highlighter.processor = FakeProcessor()
    highlighter.model = FakeModel()
    image = np.full((4, 4, 3), 100, dtype=np.uint8)

    result = highlighter.highlight(image)

    assert result.mask[0, 0] == 255
    assert result.mask[0, 1] == 0
    assert tuple(result.overlay[0, 0]) == (156, 102, 82)
    assert np.array_equal(result.overlay[0, 1], image[0, 1])


def test_mask2former_highlighter_rejects_non_rgb_images() -> None:
    highlighter = Mask2FormerRoadHighlighter.__new__(Mask2FormerRoadHighlighter)

    with pytest.raises(ValueError, match="Expected an RGB image"):
        highlighter.highlight(np.zeros((4, 4), dtype=np.uint8))
