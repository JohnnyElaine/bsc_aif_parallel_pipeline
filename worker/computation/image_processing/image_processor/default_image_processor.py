from worker.computation.image_processing.image_processor.image_processor import ImageProcessor

class DefaultImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

    def initialize(self):
        pass

    def process_image(self, img):
        return img

    def change_work_load(self, img):
        pass