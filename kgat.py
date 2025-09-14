import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, TransformerConv
from sklearn.metrics import roc_auc_score


# 搭建 KGAT 模型
class KGAT(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_heads):
        super(KGAT, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.gat_conv = GATConv(embedding_dim, embedding_dim, heads=num_heads, concat=False)
        self.predictor = nn.Linear(embedding_dim * 2, 1)

    def forward(self, edge_index, edge_type, user_indices, item_indices):
        # 获取实体和关系的嵌入
        entity_emb = self.entity_embedding(torch.arange(0, self.entity_embedding.num_embeddings).to(edge_index.device))
        relation_emb = self.relation_embedding(edge_type)

        # 知识图谱的 GAT 传播
        x = self.gat_conv(entity_emb, edge_index)

        # 获取用户和物品的嵌入
        user_emb = x[user_indices]
        item_emb = x[item_indices]

        # 拼接用户和物品嵌入，并通过预测器
        concat_emb = torch.cat([user_emb, item_emb], dim=-1)
        scores = self.predictor(concat_emb).squeeze()
        return torch.sigmoid(scores)


class KGAT3(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_heads):
        super(KGAT3, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.gat_conv1 = TransformerConv(embedding_dim, embedding_dim, heads=num_heads, dropout=0)
        self.gat_conv2 = TransformerConv(embedding_dim*num_heads, embedding_dim, heads=num_heads, dropout=0)
        # self.gat_conv3 = TransformerConv(embedding_dim*num_heads, embedding_dim, heads=num_heads, dropout=0.2)
        self.predictor = nn.Linear(embedding_dim * num_heads * 2, 1)

    def forward(self, edge_index, edge_type, user_indices, item_indices):
        # 获取实体和关系的嵌入
        entity_emb = self.entity_embedding(torch.arange(0, self.entity_embedding.num_embeddings).to(edge_index.device))
        relation_emb = self.relation_embedding(edge_type)

        # 获取边特征（例如，评分）并在消息传递中使用
        # 假设 edge_attr 是边特征矩阵，形状为 (E, D)，其中 D 是特征维度
        edge_attr = relation_emb[edge_type]  # 从边类型中获取边特征
        edge_attr = edge_attr.unsqueeze(1).expand(-1, self.num_heads, -1)
        # 知识图谱的 GAT 传播
        # 使用边特征和 TransformerConv
        x = self.gat_conv1(entity_emb, edge_index, edge_attr=edge_attr)  # 在这里使用边特征
        x = self.gat_conv2(x, edge_index, edge_attr=edge_attr)  # 在这里使用边特征
        # x = self.gat_conv3(x, edge_index, edge_attr=edge_attr)
        # x = self.gat_conv3(x, edge_index, edge_attr=edge_attr)  # 如果需要更多层

        # 获取用户和物品的嵌入
        user_emb = x[user_indices]
        item_emb = x[item_indices]

        # 拼接用户和物品嵌入，并通过预测器
        concat_emb = torch.cat([user_emb, item_emb], dim=-1)
        scores = self.predictor(concat_emb).squeeze()
        return torch.sigmoid(scores)


def load_data(kg_file, entity2id_file, relation2id_file):
    """加载知识图谱数据"""
    # 加载实体和关系的 ID 映射
    entity2id = {}
    with open(entity2id_file, 'r') as f:
        for line in f:
            entity, idx = line.strip().split('\t')
            entity2id[entity] = int(idx)
    relation2id = {}
    with open(relation2id_file, 'r') as f:
        for line in f:
            relation, idx = line.strip().split('\t')
            relation2id[relation] = int(idx)

    # 加载知识图谱三元组
    edge_index = []
    edge_type = []
    with open(kg_file, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            edge_index.append([int(head), int(tail)])
            edge_type.append(int(relation))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    return edge_index, edge_type, entity2id, relation2id


def train(model, data, optimizer, user_indices, item_indices, labels, num_epochs=50):
    """训练模型"""
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # 正向传播
        pred_scores = model(data.edge_index, data.edge_type, user_indices, item_indices)
        # 计算损失
        loss = F.binary_cross_entropy(pred_scores, labels.float())
        # 反向传播
        loss.backward()
        optimizer.step()
        # 清空日志文件，写入新的训练日志
        with open('./log/log.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}, Loss: {loss.item()}\n")

        # 打印训练日志
        if (epoch + 1) % 10 == 0:
            # 在训练集上评估模型
            auc_score = evaluate(model, data, user_indices, item_indices, labels)
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}, AUC: {auc_score}")
    # 画出损失曲线
    import matplotlib.pyplot as plt
    with open('log/log.txt', 'r') as f:
        lines = f.readlines()
        losses = [float(line.strip().split()[-1]) for line in lines]
    plt.plot(losses)


def evaluate(model, data, user_indices, item_indices, labels):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        pred_scores = model(data.edge_index, data.edge_type, user_indices, item_indices)
        auc_score = roc_auc_score(labels.cpu().numpy(), pred_scores.cpu().numpy())
    return auc_score


def main():
    # 加载数据
    edge_index, edge_type, entity2id, relation2id = load_data('kg_final.txt', 'entity2id.txt', 'relation2id.txt')
    num_entities = len(entity2id)
    num_relations = len(relation2id)

    # 构建 PyG 数据对象
    data = Data(edge_index=edge_index, edge_type=edge_type)

    # 划分训练集和测试集
    user_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 示例用户索引
    item_indices = torch.tensor([4, 5, 6, 7], dtype=torch.long)  # 示例物品索引
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)  # 示例标签

    # 初始化模型
    model = KGAT(num_entities, num_relations, embedding_dim=64, num_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    train(model, data, optimizer, user_indices, item_indices, labels, num_epochs=50)

    # 评估模型
    auc_score = evaluate(model, data, user_indices, item_indices, labels)
    print(f"Test AUC: {auc_score}")


if __name__ == '__main__':
    main()