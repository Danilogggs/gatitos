from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from app.core.config import settings
from app.ml.taxonomy import BREEDS, COAT_LENGTHS, COLORS, FEATURES, PATTERNS


@dataclass
class Prediction:
    breed: str | None
    breed_confidence: float | None
    features: list[str]
    coat_pattern: str | None
    colors: list[str]
    coat_length: str | None
    coat_confidence: float | None
    model_version: str

    def as_dict(self) -> dict:
        return vars(self)


class MLService(ABC):
    @abstractmethod
    def predict(self, images: list[bytes]) -> Prediction: ...

    @abstractmethod
    def generate_embedding(self, image: bytes) -> list[float] | None: ...


class MockMLService(MLService):
    def predict(self, images: list[bytes]) -> Prediction:
        return Prediction(None, None, [], None, [], None, None, 'mock-untrained')

    def generate_embedding(self, image: bytes) -> None:
        return None


class RealMLService(MLService):
    def __init__(self) -> None:
        import torch
        from torchvision import models, transforms

        config = settings()
        if not config.ml_model_path:
            raise RuntimeError('ML_MODEL_PATH é obrigatório em modo real')
        self.torch = torch
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = models.efficientnet_b0(weights=None)
        size = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Identity()
        self.breed_head = torch.nn.Linear(size, len(BREEDS))
        self.feature_head = torch.nn.Linear(size, len(FEATURES))
        self.pattern_head = torch.nn.Linear(size, len(PATTERNS))
        self.color_head = torch.nn.Linear(size, len(COLORS))
        self.length_head = torch.nn.Linear(size, len(COAT_LENGTHS))
        checkpoint = torch.load(config.ml_model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['backbone'])
        self.breed_head.load_state_dict(checkpoint['breed_head'])
        self.feature_head.load_state_dict(checkpoint['feature_head'])
        self.pattern_head.load_state_dict(checkpoint['pattern_head'])
        self.color_head.load_state_dict(checkpoint['color_head'])
        self.length_head.load_state_dict(checkpoint['length_head'])
        self.version = str(checkpoint.get('version', 'efficientnet-b0'))
        for module in (self.model, self.breed_head, self.feature_head, self.pattern_head, self.color_head, self.length_head):
            module.eval()

    def _features(self, image: bytes):
        with Image.open(BytesIO(image)) as opened:
            tensor = self.transform(opened.convert('RGB')).unsqueeze(0)
        with self.torch.no_grad():
            return self.model(tensor)

    def predict(self, images: list[bytes]) -> Prediction:
        with self.torch.no_grad():
            vectors = self.torch.cat([self._features(image) for image in images]).mean(dim=0, keepdim=True)
            breeds = self.torch.softmax(self.breed_head(vectors), dim=1)[0]
            feature_scores = self.torch.sigmoid(self.feature_head(vectors))[0]
            pattern_scores = self.torch.softmax(self.pattern_head(vectors), dim=1)[0]
            color_scores = self.torch.sigmoid(self.color_head(vectors))[0]
            length_scores = self.torch.softmax(self.length_head(vectors), dim=1)[0]
            breed_index = int(breeds.argmax())
            labels = list(FEATURES)
            features = [labels[i] for i, score in enumerate(feature_scores.tolist()) if score >= 0.5]
            colors = [COLORS[i] for i, score in enumerate(color_scores.tolist()) if score >= 0.5]
            pattern_index = int(pattern_scores.argmax())
            length_index = int(length_scores.argmax())
        return Prediction(BREEDS[breed_index], float(breeds[breed_index]), features,
                          PATTERNS[pattern_index], colors, COAT_LENGTHS[length_index],
                          float(pattern_scores[pattern_index]), self.version)

    def generate_embedding(self, image: bytes) -> list[float]:
        vector = self._features(image)[0]
        return self.torch.nn.functional.normalize(vector, dim=0).tolist()


def ml_service() -> MLService:
    mode = settings().ml_mode.lower()
    if mode == 'mock':
        return MockMLService()
    if mode == 'real':
        return RealMLService()
    raise RuntimeError('ML_MODE deve ser mock ou real')
