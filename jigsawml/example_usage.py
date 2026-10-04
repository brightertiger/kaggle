"""Importable examples; running this file executes the offline smoke test."""
from src.pipeline import JigsawPipeline
from src.utils.config import Config


def example_data_preparation(data_dir='data', model_dir='model'):
    pipeline = JigsawPipeline(Config(data_dir, model_dir))
    pipeline.prepare_data(data_dir)
    pipeline.generate_embeddings(data_dir)
    pipeline.generate_adversarial_data(data_dir)


def example_training(data_dir='data', model_dir='model'):
    pipeline = JigsawPipeline(Config(data_dir, model_dir))
    pipeline.train_version1(subset=0)
    pipeline.train_version2(subset=0, load_from_version1=True)


def example_scoring(data_dir='data', model_dir='model'):
    pipeline = JigsawPipeline(Config(data_dir, model_dir))
    pipeline.scoring_pipeline.score_all_models(f'{data_dir}/process/foreign/test_foreign.csv')
    return pipeline.create_ensemble()


def example_full_pipeline(data_dir='data', model_dir='model'):
    return JigsawPipeline(Config(data_dir, model_dir)).run_full_pipeline()


if __name__ == '__main__':
    from dry_run import main
    main()
