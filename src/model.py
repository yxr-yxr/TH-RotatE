import argparse
from train import train
from evaluate import evaluate
from model import THRotatE
from data_loader import KnowledgeGraphDataset
from utils import load_model
from config import Config

def main():
    parser = argparse.ArgumentParser(description="TH-RotatE: Fault Diagnosis with Knowledge Graph Embeddings")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True, help="Mode to run: train or evaluate")
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        config = Config()
        test_data = KnowledgeGraphDataset(config.test_file)
        test_loader = DataLoader(test_data, batch_size=config.test_batch_size, shuffle=False)
        
        model = THRotatE(config)
        load_model(model, config.model_save_path)
        mrr, hit_at_1, hit_at_3, hit_at_10 = evaluate(model, test_loader, config)
        print(f"Test Results - MRR: {mrr:.4f}, Hit@1: {hit_at_1:.4f}, Hit@3: {hit_at_3:.4f}, Hit@10: {hit_at_10:.4f}")

if __name__ == "__main__":
    main()
