#!/usr/bin/env python3
"""
Standalone script for training and retraining sentiment analysis models
Usage: python train_model.py --action train --dataset synthetic --bert
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_models.model_trainer import ModelTrainer
import argparse

def main():
    """Main training script"""
    
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--action', choices=['train', 'retrain', 'evaluate'], 
                       default='train', help='Action to perform')
    parser.add_argument('--dataset', choices=['imdb', 'sentiment140', 'amazon', 'synthetic'],
                       default='synthetic', help='Dataset to use for training')
    parser.add_argument('--data-path', help='Path to custom dataset CSV file')
    parser.add_argument('--bert', action='store_true', help='Fine-tune BERT model')
    parser.add_argument('--combine', action='store_true', default=True,
                       help='Combine with existing data when retraining')
    
    args = parser.parse_args()
    
    print("🧠 SentimentFusions Pro - Model Training")
    print("=" * 50)
    
    trainer = ModelTrainer()
    
    try:
        if args.action == 'train':
            print(f"🚀 Training new model on {args.dataset} dataset...")
            scores = trainer.train_new_model(args.dataset, args.bert)
            
            print("\n✅ Training completed successfully!")
            print("\n📊 Model Performance Summary:")
            for model_name, metrics in scores.items():
                print(f"  {model_name}: {metrics['test_accuracy']:.4f}")
            
        elif args.action == 'retrain':
            if not args.data_path:
                print("❌ Error: --data-path is required for retraining")
                return
            
            print(f"🔄 Retraining model with data from {args.data_path}...")
            scores = trainer.retrain_model(args.data_path, args.combine)
            
            print("\n✅ Retraining completed successfully!")
            
        elif args.action == 'evaluate':
            print("📈 Evaluating model performance...")
            accuracy, results = trainer.evaluate_model(args.data_path)
            
            print(f"\n✅ Evaluation completed!")
            print(f"📊 Model Accuracy: {accuracy:.4f}")
        
        # Export model information
        print("\n🤖 Model Information:")
        trainer.export_model_info()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())