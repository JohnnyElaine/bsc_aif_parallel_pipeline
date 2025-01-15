from worker.computation.image_processing.image_processor.image_processor import ImageProcessor

class DefaultImageProcessor(ImageProcessor):
    def process_image(self, img):
        return img

    def __init__(self):
        super().__init__()