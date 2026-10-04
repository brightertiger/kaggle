"""Python API example; expects the official files in ./data."""
from src.pipeline import CassavaPipeline
from src.utils.config import Config


def main():
    config = Config(DATA_DIR='./data', OUTPUT_DIR='./output', VERSIONS=('version7',))
    pipeline = CassavaPipeline(config)
    pipeline.prepare_data()
    for fold in range(config.N_FOLDS):
        pipeline.train_model('version7', fold)
    pipeline.score_model('version7')
    results = pipeline.create_ensemble()
    print(f'Mean blend-fold accuracy: {sum(r["model_acc"] for r in results) / len(results):.4f}')
    print('Final predictions: ./output/submission.csv')


if __name__ == '__main__':
    main()
