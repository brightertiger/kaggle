from pathlib import Path
import copy
import torch
from ..models.models import XLMRobertaClassifier
from ..data.data_utils import create_test_loader
from ..utils.config import Config
from .inference import ModelInference


class ModelScorer:
    def __init__(self, model_path, device=None, config=None):
        self.config = copy.copy(config or Config())
        self.device = device or self.config.DEVICE
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        for key, value in checkpoint.get('tokenizer_config', {}).items():
            setattr(self.config, key, value)
        model = XLMRobertaClassifier(self.config, backbone_config=checkpoint.get('backbone_config'))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.device).eval()

    def score_dataset(self, test_path, output_path):
        results = ModelInference(self.model, self.device).predict(create_test_loader(test_path, self.config))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        return results

    def score_all_folds(self, model_dir, version, test_path, output_dir):
        for fold in range(self.config.N_FOLDS):
            self.model = self.load_model(f'{model_dir}/version{version}/model_{fold}.pt')
            self.score_dataset(test_path, f'{output_dir}/version{version}/score_{fold}.csv')


class ScoringPipeline:
    def __init__(self, model_dir, config=None):
        self.model_dir = model_dir
        self.config = config or Config(model_dir=model_dir)

    def score_all_models(self, test_path):
        for version in [1, 2]:
            scorer = ModelScorer(f'{self.model_dir}/version{version}/model_0.pt', config=self.config)
            scorer.score_all_folds(self.model_dir, version, test_path, self.model_dir)
