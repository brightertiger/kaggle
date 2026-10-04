"""Compatibility entry point; all training logic lives in src.pipeline."""
from main import main
from src.pipeline import APTOSPipeline


class TrainingPipeline(APTOSPipeline):
    """Keep the legacy single-fold method names without duplicating training code."""
    def pretrain_model(self, fold: int):
        return self._pretrain_fold(fold)

    def train_model(self, fold: int):
        return self._train_fold(fold)

    def combine_training(self, fold: int | None = None):
        if fold is None:
            return super().combine_training()
        return self._combine_fold(fold)


if __name__ == '__main__':
    main()
