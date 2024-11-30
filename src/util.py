import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")

def calculate_metrics(scores, labels):
    # 将所有分数和标签拼接
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # 计算MRR和Hit@K
    mrr = (labels / scores).mean().item()
    hit_at_1 = (labels[scores.topk(1, dim=-1).indices]).mean().item()
    hit_at_3 = (labels[scores.topk(3, dim=-1).indices]).mean().item()
    hit_at_10 = (labels[scores.topk(10, dim=-1).indices]).mean().item()
    
    return mrr, hit_at_1, hit_at_3, hit_at_10

def log_metrics(step, mrr, hit_at_1, hit_at_3, hit_at_10):
    print(f"Step {step}: MRR={mrr:.4f}, Hit@1={hit_at_1:.4f}, Hit@3={hit_at_3:.4f}, Hit@10={hit_at_10:.4f}")
