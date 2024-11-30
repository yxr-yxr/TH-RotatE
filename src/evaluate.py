import torch
from utils import calculate_metrics

def evaluate(model, data_loader, config):
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            positive_sample, negative_sample, labels = batch
            if torch.cuda.is_available():
                positive_sample = positive_sample.cuda()
                negative_sample = negative_sample.cuda()
                labels = labels.cuda()
            
            scores = model.predict(positive_sample, negative_sample)
            all_scores.append(scores)
            all_labels.append(labels)
    
    # 计算MRR和Hit@K
    mrr, hit_at_1, hit_at_3, hit_at_10 = calculate_metrics(all_scores, all_labels)
    return mrr, hit_at_1, hit_at_3, hit_at_10
