from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

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
    breed_scores: dict[str, float] = field(default_factory=dict)

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
        model_path = Path(config.ml_model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[2] / model_path
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['backbone'])
        self.breeds = list(checkpoint.get('breeds', BREEDS))
        if not self.breeds or len(set(self.breeds)) != len(self.breeds) or set(self.breeds) - set(BREEDS):
            raise ValueError('Checkpoint contém raças incompatíveis com a taxonomia')
        trained_heads = set(checkpoint.get('trained_heads', ['breed', 'feature', 'pattern', 'color', 'length']))

        def load_head(name: str, classes: int):
            if name not in trained_heads:
                return None
            head = torch.nn.Linear(size, classes)
            head.load_state_dict(checkpoint[f'{name}_head'])
            head.eval()
            return head

        self.breed_head = load_head('breed', len(self.breeds))
        if self.breed_head is None:
            raise ValueError('Checkpoint sem cabeça de raça treinada')
        self.feature_head = load_head('feature', len(FEATURES))
        self.pattern_head = load_head('pattern', len(PATTERNS))
        self.color_head = load_head('color', len(COLORS))
        self.length_head = load_head('length', len(COAT_LENGTHS))
        self.breed_min_confidence = config.ml_breed_min_confidence
        self.version = str(checkpoint.get('version', 'efficientnet-b0'))
        self.model.eval()

    def _features(self, image: bytes):
        with Image.open(BytesIO(image)) as opened:
            tensor = self.transform(opened.convert('RGB')).unsqueeze(0)
        with self.torch.no_grad():
            return self.model(tensor)

    def predict(self, images: list[bytes]) -> Prediction:
        with self.torch.no_grad():
            image_vectors = self.torch.cat([self._features(image) for image in images])
            vectors = image_vectors.mean(dim=0, keepdim=True)
            breeds = self.torch.softmax(self.breed_head(vectors), dim=1)[0]
            breed_index = int(breeds.argmax())
            breed_confidence = float(breeds[breed_index])
            breed = self.breeds[breed_index]
            # A maior pontuação vence; SRD é provisório quando nenhuma classe passa do limite.
            if 'srd' not in self.breeds and breed_confidence <= self.breed_min_confidence:
                breed = 'srd'
                breed_confidence = None  # A probabilidade da raça rejeitada não mede confiança em SRD.
            features = []
            if self.feature_head is not None:
                scores = self.torch.sigmoid(self.feature_head(vectors))[0]
                features = [label for label, score in zip(FEATURES, scores.tolist()) if score >= 0.5]
            colors = []
            if self.color_head is not None:
                scores = self.torch.sigmoid(self.color_head(vectors))[0]
                colors = [label for label, score in zip(COLORS, scores.tolist()) if score >= 0.5]
            coat_pattern = None
            coat_confidence = None
            if self.pattern_head is not None:
                scores = self.torch.softmax(self.pattern_head(vectors), dim=1)[0]
                index = int(scores.argmax())
                coat_pattern = PATTERNS[index]
                coat_confidence = float(scores[index])
            coat_length = None
            if self.length_head is not None:
                scores = self.torch.softmax(self.length_head(vectors), dim=1)[0]
                coat_length = COAT_LENGTHS[int(scores.argmax())]
        return Prediction(breed, breed_confidence, features,
                          coat_pattern, colors, coat_length, coat_confidence, self.version,
                          {label: float(score) for label, score in zip(self.breeds, breeds.tolist())})

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
