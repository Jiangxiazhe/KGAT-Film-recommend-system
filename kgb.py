import ast
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from kgat import *
import data_process as dp


# 1. 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")


# 2. 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"模型已从 {path} 加载")
    return model


# 3. 测试函数
def test(model, data, user_indices, item_indices, labels):
    model.eval()
    with torch.no_grad():
        pred_scores = model(data.edge_index, data.edge_type, user_indices, item_indices)
        auc_score = roc_auc_score(labels.cpu().numpy(), pred_scores.cpu().numpy())
    return auc_score


# 4. 主函数
def main():
    # 加载数据，如果已经划分过数据集，可以直接加载
    if os.path.exists('./pre_data/train_ratings.csv'):
        train_ratings_for_kg = pd.read_csv('./pre_data/train_ratings.csv')
        test_ratings = pd.read_csv('./pre_data/test_ratings.csv')
        c_user_ratings = pd.read_csv('./pre_data/cold_start_ratings.csv')
        users = pd.read_csv('./pre_data/users.csv')
        movies = pd.read_csv('./pre_data/movies.csv')
    else:
        train_ratings_for_kg, test_ratings, c_user_ratings, users, movies = dp.dataset_split()
    # 构建知识图谱（仅包含训练集评分），如果已经构建过知识图谱，可以直接加载
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
    else:
        triples, entity2id, relation2id = dp.build_kg_graph(train_ratings_for_kg, users, movies)

    # 准备训练数据
    user_indices, item_indices, labels = dp.prepare_data(train_ratings_for_kg, entity2id)

    # 构建 PyG 的 Data 对象
    edge_index = torch.tensor([[triple[0] for triple in triples], [triple[2] for triple in triples]], dtype=torch.long)
    edge_type = torch.tensor([triple[1] for triple in triples], dtype=torch.long)
    data = Data(edge_index=edge_index, edge_type=edge_type)

    # # 初始化模型
    # entitys_len = {}
    # # 获取每种实体的数量
    # for entity in entity2id.keys():
    #     if "User" in entity:
    #         entitys_len["User"] = entitys_len.get("User", 0) + 1
    #     elif "Movie" in entity:
    #         entitys_len["Movie"] = entitys_len.get("Movie", 0) + 1
    #     elif "Age_group" in entity:
    #         entitys_len["Age_group"] = entitys_len.get("Age_group", 0) + 1
    #     elif "Gender" in entity:
    #         entitys_len["Gender"] = entitys_len.get("Gender", 0) + 1
    #     elif "Zipcode" in entity:
    #         entitys_len["Zipcode"] = entitys_len.get("Zipcode", 0) + 1
    #     elif "Year" in entity:
    #         entitys_len["Year"] = entitys_len.get("Year", 0) + 1
    #     elif "Genre" in entity:
    #         entitys_len["Genre"] = entitys_len.get("Genre", 0) + 1
    #     elif "Occupation" in entity:
    #         entitys_len["Occupation"] = entitys_len.get("Occupation", 0) + 1

    # model = KGAT(num_entities=len(entity2id), num_relations=len(relation2id), embedding_dim=32, num_heads=4)
    model = KGAT3(num_entities=len(entity2id), num_relations=len(relation2id), embedding_dim=16, num_heads=4)
    # model = KGAT2(num_entities=len(entity2id), num_relations=len(relation2id), embedding_dim=32, num_heads=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 获得用户特征和物品特征
    # user_features = dp.extract_user_features(users, embedding_dim=16)
    # item_features = dp.extract_item_features(movies, embedding_dim=16)

    # 训练模型
    train(model, data, optimizer, user_indices, item_indices, labels, num_epochs=200)

    # 保存模型
    save_model(model, './model/kgat_model.pth')

    # 加载模型
    model = load_model(model, './model/kgat_model.pth')

    # 在测试集上评估
    test_user_indices, test_item_indices, test_labels = dp.prepare_data(test_ratings, entity2id)
    test_auc = test(model, data, test_user_indices, test_item_indices, test_labels)
    print(f"测试集 AUC: {test_auc}")

    # 在冷启动用户集上评估
    cold_user_indices, cold_item_indices, cold_labels = dp.prepare_data(c_user_ratings, entity2id)
    cold_auc = test(model, data, cold_user_indices, cold_item_indices, cold_labels)
    print(f"冷启动用户集 AUC: {cold_auc}")

    # 修改模型名称
    new_model_name = f'./model/kgat_model_{int(1000*test_auc)}.pth'
    save_model(model, new_model_name)

if __name__ == '__main__':
    main()