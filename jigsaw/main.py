#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.pipeline import JigsawPipeline


def main():
    """Main entry point for Jigsaw Toxic Comment Classification."""
    parser = argparse.ArgumentParser(
        description="Jigsaw Toxic Comment Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with both BERT and GPT models
  python main.py --step full --model-type both

  # Process data only
  python main.py --step process-data

  # Train BERT models only
  python main.py --step train --model-type bert

  # Generate test predictions
  python main.py --step predict --model-type bert

  # Evaluate models
  python main.py --step evaluate

  # Custom data path
  python main.py --step full --data-path /path/to/data --output-path /path/to/output
        """
    )
    
    parser.add_argument(
        '--step',
        choices=['process-data', 'train', 'evaluate', 'predict', 'full'],
        required=True,
        help='Pipeline step to execute'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['bert', 'gpt', 'both'],
        default='both',
        help='Model type to use (default: both)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data',
        help='Path to data directory (default: ../data)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='../model',
        help='Path to model directory (default: ../model)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='../output',
        help='Path to output directory (default: ../output)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use for training (default: cuda:0)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=222,
        help='Maximum sequence length (default: 222)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=12,
        help='Training batch size (default: 12)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("🚀 Jigsaw Toxic Comment Classification Pipeline")
    print("=" * 60)
    print(f"Step: {args.step}")
    print(f"Model Type: {args.model_type}")
    print(f"Data Path: {args.data_path}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Device: {args.device}")
    print(f"Random Seed: {args.random_seed}")
    print("=" * 60)
    
    try:
        config = Config()
        config.data_path = args.data_path
        config.model_path = args.model_path
        config.output_path = args.output_path
        config.device = args.device
        config.random_seed = args.random_seed
        config.n_folds = args.n_folds
        config.max_length = args.max_length
        
        config.bert_config.update({
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs
        })
        
        config.gpt_config.update({
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs
        })
        
        pipeline = JigsawPipeline(config)
        
        if args.step == 'process-data':
            print("📊 Processing data...")
            pipeline.process_data()
            print("✅ Data processing completed")
        
        elif args.step == 'train':
            print(f"🤖 Training {args.model_type} models...")
            pipeline.train_all_models(args.model_type)
            print("✅ Training completed")
        
        elif args.step == 'evaluate':
            print("📈 Evaluating models...")
            results = pipeline.evaluate_models()
            
            print("\n📊 Evaluation Results:")
            print("-" * 40)
            for fold_name, fold_results in results.items():
                print(f"\n{fold_name}:")
                for model_name, metrics in fold_results.items():
                    print(f"  {model_name}:")
                    print(f"    Final Metric: {metrics['final_metric']:.5f}")
                    print(f"    Overall AUC: {metrics['overall_auc']:.5f}")
                    print(f"    Subgroup AUC: {metrics['subgroup_auc_mean']:.5f}")
            
            print("✅ Evaluation completed")
        
        elif args.step == 'predict':
            print(f"🔮 Generating predictions with {args.model_type} models...")
            predictions = pipeline.generate_test_predictions(args.model_type)
            if not predictions.empty:
                print(f"✅ Generated {len(predictions)} predictions")
            else:
                print("❌ No predictions generated")
        
        elif args.step == 'full':
            print("🚀 Running full pipeline...")
            results = pipeline.run_full_pipeline(args.model_type)
            
            print("\n🎉 Pipeline completed successfully!")
            print("\n📊 Summary:")
            print("-" * 40)
            
            if 'evaluations' in results:
                for fold_name, fold_results in results['evaluations'].items():
                    print(f"\n{fold_name}:")
                    for model_name, metrics in fold_results.items():
                        print(f"  {model_name} - Final Metric: {metrics['final_metric']:.5f}")
            
            if 'test_predictions' in results:
                for model_name, pred_df in results['test_predictions'].items():
                    if not pred_df.empty:
                        print(f"  {model_name} - Generated {len(pred_df)} test predictions")
    
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
