import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

# TransH模型
class TransH(nn.Module):
    def __init__(self, entity_num, relation_num, hidden_dim):
        super(TransH, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.hidden_dim = hidden_dim

        # Embedding初始化：实体、关系
        self.entity_embedding = nn.Embedding(entity_num, hidden_dim)
        self.relation_embedding = nn.Embedding(relation_num, hidden_dim)
        
        # 平面法向量，用于TransH
        self.normal_vector = nn.Embedding(relation_num, hidden_dim)

        self.init_weights()

    def init_weights(self):
        # 初始化权重
        init.xavier_uniform_(self.entity_embedding.weight)
        init.xavier_uniform_(self.relation_embedding.weight)
        init.xavier_uniform_(self.normal_vector.weight)

    def forward(self, head, relation, tail):
        # TransH：实体嵌入与关系的法向量之间的投影
        head_embedding = self.entity_embedding(head)
        tail_embedding = self.entity_embedding(tail)
        relation_embedding = self.relation_embedding(relation)
        normal_vector = self.normal_vector(relation)

        # 计算TransH模型中的“投影”
        head_proj = head_embedding - torch.sum(head_embedding * normal_vector, dim=-1, keepdim=True) * normal_vector
        tail_proj = tail_embedding - torch.sum(tail_embedding * normal_vector, dim=-1, keepdim=True) * normal_vector

        # 损失计算（基于L2距离）
        score = torch.norm(head_proj + relation_embedding - tail_proj, p=2, dim=-1)
        return score

# RotatE模型
class RotatE(nn.Module):
    def __init__(self, entity_num, relation_num, hidden_dim):
        super(RotatE, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.hidden_dim = hidden_dim

        # 实体和关系的嵌入
        self.entity_embedding = nn.Embedding(entity_num, hidden_dim)
        self.relation_embedding = nn.Embedding(relation_num, hidden_dim)
        
        self.init_weights()

    def init_weights(self):
        # 初始化权重
        init.xavier_uniform_(self.entity_embedding.weight)
        init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, head, relation, tail):
        # RotatE：计算复数乘法的旋转变换
        head_embedding = self.entity_embedding(head)
        tail_embedding = self.entity_embedding(tail)
        relation_embedding = self.relation_embedding(relation)

        # 计算旋转
        rotated_head = head_embedding * torch.cos(relation_embedding) + head_embedding * torch.sin(relation_embedding)
        rotated_tail = tail_embedding

        # 损失计算（基于L2距离）
        score = torch.norm(rotated_head + relation_embedding - rotated_tail, p=2, dim=-1)
        return score

# TH-RotatE 模型：结合TransH和RotatE
class THRotatE(nn.Module):
    def __init__(self, entity_num, relation_num, hidden_dim):
        super(THRotatE, self).__init__()
        
        # 分别初始化TransH和RotatE部分
        self.transh_model = TransH(entity_num, relation_num, hidden_dim)
        self.rotate_model = RotatE(entity_num, relation_num, hidden_dim)

    def forward(self, head, relation, tail):
        # 计算TransH部分的得分
        transh_score = self.transh_model(head, relation, tail)

        # 计算RotatE部分的得分
        rotate_score = self.rotate_model(head, relation, tail)

        # 融合TransH和RotatE的得分
        final_score = transh_score + rotate_score

        return final_score

# 自定义损失函数（典型的排名损失）
class MarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_score, negative_score):
        # 根据margin loss计算正样本和负样本的损失
        loss = torch.max(positive_score - negative_score + self.margin, torch.zeros_like(positive_score))
        return torch.mean(loss)

