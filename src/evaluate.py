import torch
from torch.utils.data import DataLoader
from model import TH_RotatE
from data_loader import CustomDataset
from utils import compute_metrics
import argparse
import os

# 设定测试函数
def evaluate(model, dataloader, device):
    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0
    total = 0
    metrics = {
        'MRR': 0.0,
        'Hit@1': 0.0,
        'Hit@3': 0.0,
        'Hit@10': 0.0,
    }

    with torch.no_grad():  # 在评估时禁用梯度计算
        for batch in dataloader:
            # 从数据加载器获取批次数据
            input_data, target = batch
            input_data = input_data.to(device)
            target = target.to(device)

            # 前向传播
            output = model(input_data)

            # 计算损失
            loss = model.loss(output, target)
            total_loss += loss.item()

            # 计算评估指标
            batch_metrics = compute_metrics(output, target)
            for metric in metrics:
                metrics[metric] += batch_metrics[metric]

            # 计算分类准确率
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # 计算所有指标的平均值
    metrics = {metric: value / len(dataloader) for metric, value in metrics.items()}
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, metrics

# 主函数
def main():
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to the test dataset.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument('--device', type=str, default='cuda', help="Device for evaluation (cpu or cuda).")
    args = parser.parse_args()

    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = TH_RotatE()  # 根据实际情况修改模型的初始化方式
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # 加载测试数据集
    test_dataset = CustomDataset(args.test_data)  # 使用自定义数据集类
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 开始评估
    avg_loss, accuracy, metrics = evaluate(model, test_dataloader, device)

    # 打印结果
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
