import torch
import pandas as pd
from torch_geometric.data import Data
from kgat import *


# 加载模型
def load_model(model_path, num_entities, num_relations, embedding_dim, num_heads):
    model = KGAT3(num_entities, num_relations, embedding_dim, num_heads)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 加载数据
def load_data():
    users = pd.read_csv('./ml-1m/users.dat', sep='::', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies = pd.read_csv('./ml-1m/movies.dat', sep='::', encoding='ISO-8859-1', names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    return users, movies, ratings

# 加载实体和关系映射
def load_mappings(entity2id_file, relation2id_file):
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
    return entity2id, relation2id


def get_unseen_movies(user_id, ratings, movies):
    # 获取用户已看过的电影
    seen_movies = set(ratings[ratings['UserID'] == user_id]['MovieID'])
    # 获取所有电影
    all_movies = set(movies['MovieID'])
    # 返回未看过的电影
    return list(all_movies - seen_movies)


def recommend_top_k(model, data, user_id, unseen_movies, entity2id, k=10):
    # 将用户 ID 和未看过的电影 ID 转换为实体索引
    user_idx = entity2id[f"User{user_id}"]
    movie_indices = [entity2id[f"Movie{movie_id}"] for movie_id in unseen_movies]

    # 构建输入数据
    user_indices = torch.tensor([user_idx] * len(movie_indices), dtype=torch.long)
    item_indices = torch.tensor(movie_indices, dtype=torch.long)

    # 预测评分
    with torch.no_grad():
        pred_scores = model(data.edge_index, data.edge_type, user_indices, item_indices)

    # 将预测评分与电影 ID 结合
    movie_scores = list(zip(unseen_movies, pred_scores.cpu().numpy()))

    # 按评分排序，取 Top-K
    movie_scores_sorted = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:k]

    return movie_scores_sorted

def format_recommendations(movie_scores, movies):
    recommendations = []
    for movie_id, score in movie_scores:
        movie_info = movies[movies['MovieID'] == movie_id].iloc[0]
        recommendations.append({
            'MovieID': movie_id,
            'Title': movie_info['Title'],
            'Genres': movie_info['Genres'],
            'PredictedScore': score
        })
    return recommendations


# 职业映射表
OCCUPATION_MAP = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}


def main(user_id, k=10):
    # 加载模型和数据
    users, movies, ratings = load_data()
    entity2id, relation2id = load_mappings('entity2id.txt', 'relation2id.txt')
    model = load_model('./model/kgat_model.pth', num_entities=len(entity2id), num_relations=len(relation2id), embedding_dim=16,
                       num_heads=4)

    # 读取并构建 PyG 的 Data 对象
    triples = []
    with open('kg_final.txt', 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triples.append((int(head), int(relation), int(tail)))
    edge_index = torch.tensor([[triple[0] for triple in triples], [triple[2] for triple in triples]], dtype=torch.long)
    edge_type = torch.tensor([triple[1] for triple in triples], dtype=torch.long)
    data = Data(edge_index=edge_index, edge_type=edge_type)
    # 输出用户的性别、年龄、职业等信息
    user_info = users[users['UserID'] == user_id].iloc[0]
    print(f"用户 {user_id} 的信息：")
    print(f"性别: {user_info['Gender']}\t年龄: {user_info['Age']}\t职业: {OCCUPATION_MAP[user_info['Occupation']]}")

    # 获取用户未看过的电影
    unseen_movies = get_unseen_movies(user_id, ratings, movies)

    # 生成推荐列表
    movie_scores = recommend_top_k(model, data, user_id, unseen_movies, entity2id, k)

    # 格式化推荐结果
    recommendations = format_recommendations(movie_scores, movies)

    # 输出推荐结果
    print(f"为用户 {user_id} 推荐的 Top-{k} 电影：")
    for rec in recommendations:
        print(f"电影ID: {rec['MovieID']}, 标题: {rec['Title']}, 类型: {rec['Genres']}, 预测评分: {rec['PredictedScore']:.4f}")


if __name__ == '__main__':
    user_id = 1  # 输入用户 ID
    user_id = int(input("用户ID："))
    k = 10  # 推荐数量
    k = int(input("推荐电影数量："))
    main(user_id, k)
