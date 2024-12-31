import numpy as np

from aif_edge_node.image_processing.image_processor.image_processor import ImageProcessor


class DefaultmageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

    def _draw_bounding_boxes(self, image, bounding_boxes):
        pass

    def _extract_bounding_boxes(self, inference_result) -> np.ndarray:
        pass



