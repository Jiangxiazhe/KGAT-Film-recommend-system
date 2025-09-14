
import os

import numpy as np
import pandas as pd

from kgat import *

# 2. 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"模型已从 {path} 加载")
    return model


def prepare_data(train_ratings, entity2id):
    # 正样本：训练集中的用户-电影对
    user_indices = [entity2id[f"User{uid}"] for uid in train_ratings['UserID']]
    item_indices = [entity2id[f"Movie{mid}"] for mid in train_ratings['MovieID']]
    labels = torch.tensor(train_ratings['Rating'].apply(lambda x: 1 if x >= 3 else 0).values, dtype=torch.float)
    # labels中正负样本的数量
    print(f"正样本数量: {labels.sum()}, 负样本数量: {len(labels) - labels.sum()}")
    # 转换为Tensor
    user_indices = torch.tensor(user_indices, dtype=torch.long)
    item_indices = torch.tensor(item_indices, dtype=torch.long)
    return user_indices, item_indices, labels


def test(model, data, user_indices, item_indices, labels, k=10):
    model.eval()
    with torch.no_grad():
        # 获取模型预测的评分
        pred_scores = model(data.edge_index, data.edge_type, user_indices, item_indices)

        # 计算 AUC
        auc_score = roc_auc_score(labels.cpu().numpy(), pred_scores.cpu().numpy())

        # 按用户分组
        user_to_preds = {}
        user_to_labels = {}
        for user_idx, item_idx, pred_score, label in zip(user_indices, item_indices, pred_scores, labels):
            user_idx = user_idx.item()
            item_idx = item_idx.item()
            pred_score = pred_score.item()
            label = label.item()

            if user_idx not in user_to_preds:
                user_to_preds[user_idx] = []
                user_to_labels[user_idx] = []
            user_to_preds[user_idx].append((item_idx, pred_score))
            user_to_labels[user_idx].append((item_idx, label))

        # 初始化 Precision@K 和 Recall@K 的列表
        precisions = []
        recalls = []

        # 遍历每个用户，计算 Precision@K 和 Recall@K
        for user_idx in user_to_preds:
            # 获取当前用户的预测评分和真实评分
            user_preds = user_to_preds[user_idx]
            user_labels = user_to_labels[user_idx]

            # 按预测评分排序，取前 K 个物品
            user_preds_sorted = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
            top_k_items = [item[0] for item in user_preds_sorted]

            # 获取真实标签
            true_items = set(item[0] for item in user_labels if item[1] == 1)  # 假设标签为 1 表示正样本
            recommended_items = set(top_k_items)

            # 计算 Precision@K，考虑到k可能大于用户的正样本数，所以需要取min(k, len(true_items))
            precision = len(true_items & recommended_items) / min(k, len(recommended_items))
            precisions.append(precision)

            # 计算 Recall@K
            recall = len(true_items & recommended_items) / len(true_items) if len(true_items) > 0 else 0
            recalls.append(recall)

        # 计算平均 Precision@K 和 Recall@K
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

    return auc_score, avg_precision, avg_recall


# 读取entity等的映射文件
triples = []
entity2id = {}
relation2id = {}
if os.path.exists('kg_final.txt'):
    with open('kg_final.txt', 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triples.append((int(head), int(relation), int(tail)))
    with open('entity2id.txt', 'r') as f:
        for line in f:
            entity, idx = line.strip().split('\t')
            entity2id[entity] = int(idx)
    with open('relation2id.txt', 'r') as f:
        for line in f:
            relation, idx = line.strip().split('\t')
            relation2id[relation] = int(idx)


# 加载模型
# model = KGAT2(num_entities=len(entity2id), num_relations=len(relation2id), embedding_dim=32, num_heads=4)
model = KGAT3(num_entities=len(entity2id), num_relations=len(relation2id), embedding_dim=16, num_heads=4)
model = load_model(model, './model/kgat_model.pth')

# 构建 PyG 的 Data 对象
edge_index = torch.tensor([[triple[0] for triple in triples], [triple[2] for triple in triples]], dtype=torch.long)
edge_type = torch.tensor([triple[1] for triple in triples], dtype=torch.long)
data = Data(edge_index=edge_index, edge_type=edge_type)
# 读取测试集评分数据
test_ratings = pd.read_csv('./pre_data/test_ratings.csv')
# 读取冷启动用户评分数据
c_user_ratings = pd.read_csv('./pre_data/cold_start_ratings.csv')

# 在测试集上评估
test_user_indices, test_item_indices, test_labels = prepare_data(test_ratings, entity2id)
test_auc, test_avg_pre, test_avg_recall = test(model, data, test_user_indices, test_item_indices, test_labels)
print(f"测试集 AUC: {test_auc},Pre:{test_avg_pre},Recall:{test_avg_recall}")

# 在冷启动用户集上评估
cold_user_indices, cold_item_indices, cold_labels = prepare_data(c_user_ratings, entity2id)
cold_auc, cold_avg_pre, cold_avg_recall = test(model, data, cold_user_indices, cold_item_indices, cold_labels)
print(f"冷启动用户集 AUC: {cold_auc},Pre:{cold_avg_pre},Recall:{cold_avg_recall}")

# 计算不同K值下的Precision@K和Recall@K
k_values = range(1, 300, 5)
precisions = []
recalls = []
for k in k_values:
    test_auc, test_avg_pre, test_avg_recall = test(model, data, test_user_indices, test_item_indices, test_labels, k)
    cold_auc, cold_avg_pre, cold_avg_recall = test(model, data, cold_user_indices, cold_item_indices, cold_labels, k)
    precisions.append((test_avg_pre, cold_avg_pre))
    recalls.append((test_avg_recall, cold_avg_recall))

# 画出Precision@K和Recall@K曲线
import matplotlib.pyplot as plt
plt.plot(k_values, [p[0] for p in precisions], label='Test Precision')
plt.plot(k_values, [p[1] for p in precisions], label='Cold Start Precision')
plt.plot(k_values, [r[0] for r in recalls], label='Test Recall')
plt.plot(k_values, [r[1] for r in recalls], label='Cold Start Recall')
plt.xlabel('K')
plt.ylabel('Value')
plt.legend()
plt.show()


