import random

class Instance:
    def __init__(self, id_, embedding, bbox, last_frame):
        self.id_ = id_
        self.embedding = embedding
        self.bbox = bbox
        self.last_frame = last_frame

        random.seed(self.id_)
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
