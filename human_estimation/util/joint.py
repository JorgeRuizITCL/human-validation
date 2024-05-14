from attrs import define


@define
class Joint:
    id: int  # Joint as it appeared in the inference result
    x: float  # x coordinate, 0-1 relative to image width
    y: float  # y coordinate, 0-1 relative to image height
    label: str  # Joint label
    confidence: float  # Prediction Confidence (0,1)
    threshold: float  # Threshold for visibility

    @property
    def is_visible(self) -> bool:
        return self.confidence > self.threshold
