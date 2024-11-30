import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import THRotatE
from data_loader import KnowledgeGraphDataset
from utils import save_model, log_metrics
from config import Config

def train():
    # 加载配置
    config = Config()
    
    # 加载数据
    train_data = KnowledgeGraphDataset(config.train_file)
    valid_data = KnowledgeGraphDataset(config.valid_file)
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.test_batch_size, shuffle=False)
    
    # 初始化模型
    model = THRotatE(config)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCELoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 训练循环
    best_mrr = 0
    for step in range(config.max_steps):
        model.train()
        total_loss = 0
        for batch in train_loader:
            positive_sample, negative_sample, labels = batch
            
            if torch.cuda.is_available():
                positive_sample = positive_sample.cuda()
                negative_sample = negative_sample.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            loss = model(positive_sample, negative_sample, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 评估模型
        if step % config.eval_steps == 0:
            model.eval()
            mrr, hit_at_1, hit_at_3, hit_at_10 = evaluate(model, valid_loader, config)
            log_metrics(step, mrr, hit_at_1, hit_at_3, hit_at_10)
            
            # 保存最优模型
            if mrr > best_mrr:
                best_mrr = mrr
                save_model(model, config.model_save_path)
        
        print(f"Step {step}/{config.max_steps}, Loss: {total_loss:.4f}")
    
    print("Training completed.")

if __name__ == "__main__":
    train()
