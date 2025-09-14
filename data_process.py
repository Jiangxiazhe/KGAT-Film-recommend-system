# 读取数据
import ast

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn


def load_data():
    users = pd.read_csv('./ml-1m/users.dat', sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies = pd.read_csv('./ml-1m/movies.dat', encoding='ISO-8859-1', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    return users, movies, ratings

# 定义年龄分组
def age_to_group(age):
    if age < 18:
        return '0-18'
    elif age < 25:
        return '18-25'
    elif age < 35:
        return '25-35'
    elif age < 45:
        return '35-45'
    elif age < 56:
        return '45-55'
    else:
        return '56+'

# 初始化实体和关系的 ID 映射
entity2id = {}
relation2id = {
    'rate': 0,
    'belong_to': 1,
    'has_occupation': 2,
    'in_age_group': 3,
    'has_gender': 4,
    'lives_in': 5,
    'released_in': 6
}

# 生成实体到 ID 的映射
def get_entity_id(entity):
    if entity not in entity2id:
        entity2id[entity] = len(entity2id)
    return entity2id[entity]

def dataset_split():
    users, movies, ratings = load_data()
    # 处理用户数据
    users['Age_group'] = users['Age'].apply(age_to_group)

    # 处理电影数据：将类型拆分为列表
    movies['Genres'] = movies['Genres'].apply(lambda x: x.split('|'))
    # 划分训练集、验证集和测试集
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    # 冷启动用户集划分
    cold_start_users = users.sample(frac=0.1, random_state=42)['UserID'].tolist()
    cold_start_ratings = ratings[ratings['UserID'].isin(cold_start_users)]

    # 将冷启动数据从训练集中移除
    train_ratings = train_ratings[~train_ratings['UserID'].isin(cold_start_users)]

    # 保存划分后的数据
    train_ratings.to_csv('./pre_data/train_ratings.csv', index=False)
    test_ratings.to_csv('./pre_data/test_ratings.csv', index=False)
    cold_start_ratings.to_csv('./pre_data/cold_start_ratings.csv', index=False)
    users.to_csv('./pre_data/users.csv', index=False)
    movies.to_csv('./pre_data/movies.csv', index=False)

    return train_ratings, test_ratings, cold_start_ratings, users, movies


# 生成知识图谱三元组
def build_kg_graph(ratings, users, movies):
    triples = []

    # 1. 用户-电影评分关系
    for _, row in ratings.iterrows():
        user_entity = f"User{row['UserID']}"
        movie_entity = f"Movie{row['MovieID']}"
        user_id = get_entity_id(user_entity, 'User')
        movie_id = get_entity_id(movie_entity, 'Movie')
        triples.append((user_id, relation2id['rate'], movie_id))

    # 2. 电影-类型关系
    for _, row in movies.iterrows():
        movie_entity = f"Movie{row['MovieID']}"
        movie_id = get_entity_id(movie_entity, 'Movie')
        # 将Str的类型转为List，例子：""['Animation', ""Children's"", 'Comedy']""
        Genres_str = row['Genres']
        Genres = Genres_str.strip('"')
        Genres = ast.literal_eval(Genres)
        for genre in Genres:
            genre_entity = f'Genre{genre}'
            genre_id = get_entity_id(genre_entity, 'Genre')
            triples.append((movie_id, relation2id['belong_to'], genre_id))

    # 3. 用户-职业关系
    for _, row in users.iterrows():
        user_entity = f"User{row['UserID']}"
        user_id = get_entity_id(user_entity, 'User')
        occupation_entity = f'Occupation{row["Occupation"]}'
        occupation_id = get_entity_id(occupation_entity, 'Occupation')
        triples.append((user_id, relation2id['has_occupation'], occupation_id))

    # 4. 用户-年龄组关系
    for _, row in users.iterrows():
        user_entity = f"User{row['UserID']}"
        user_id = get_entity_id(user_entity, 'User')
        age_group_entity = f'Age_group{row["Age_group"]}'
        age_group_id = get_entity_id(age_group_entity, 'Age_group')
        triples.append((user_id, relation2id['in_age_group'], age_group_id))

    # 5. 用户-性别关系
    for _, row in users.iterrows():
        user_entity = f"User{row['UserID']}"
        user_id = get_entity_id(user_entity, 'User')
        gender_entity = f"Gender{row['Gender']}"
        gender_id = get_entity_id(gender_entity, 'Gender')
        triples.append((user_id, relation2id['has_gender'], gender_id))

    # 6. 用户-邮编关系
    for _, row in users.iterrows():
        user_entity = f"User{row['UserID']}"
        user_id = get_entity_id(user_entity, 'User')
        zip_code_entity = f"Zipcode{row['Zip-code']}"
        zip_code_id = get_entity_id(zip_code_entity, 'Zipcode')
        triples.append((user_id, relation2id['lives_in'], zip_code_id))

    # 7. 电影-年份关系（假设我们使用正则从电影标题中提取年份）Title (year)
    movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)')[0]

    for _, row in movies.iterrows():
        movie_entity = f"Movie{row['MovieID']}"
        movie_id = get_entity_id(movie_entity, 'Movie')
        year_entity = f"Year{str(row['Year'])}"
        year_id = get_entity_id(year_entity, 'Year')
        triples.append((movie_id, relation2id['released_in'], year_id))

    # 保存知识图谱三元组
    with open('kg_final.txt', 'w') as f:
        for triple in triples:
            f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

    # 保存实体到 ID 的映射
    with open('entity2id.txt', 'w') as f:
        for entity, idx in entity2id.items():
            f.write(f"{entity}\t{idx}\n")

    # 保存关系到 ID 的映射
    with open('relation2id.txt', 'w') as f:
        for relation, idx in relation2id.items():
            f.write(f"{relation}\t{idx}\n")

    print("知识图谱构建完成！")
    print(f"实体数量: {len(entity2id)}")
    print(f"关系数量: {len(relation2id)}")
    print(f"三元组数量: {len(triples)}")

    return triples, entity2id, relation2id


def prepare_data(train_ratings, entity2id):
    # 正样本：训练集中的用户-电影对
    user_indices = [entity2id[f"User{uid}"] for uid in train_ratings['UserID']]
    item_indices = [entity2id[f"Movie{mid}"] for mid in train_ratings['MovieID']]
    labels = torch.tensor(train_ratings['Rating'].apply(lambda x: 1 if x >= 3 else 0).values, dtype=torch.float)

    # 转换为Tensor
    user_indices = torch.tensor(user_indices, dtype=torch.long)
    item_indices = torch.tensor(item_indices, dtype=torch.long)
    return user_indices, item_indices, labels


# 用户特征提取
def extract_user_features(users, embedding_dim=32, user_emb_size=724442):
    # 职业嵌入
    num_occupations = users['Occupation'].nunique()
    occupation_embedding = nn.Embedding(num_occupations, embedding_dim)
    occupation_features = occupation_embedding(torch.tensor(users['Occupation'].values))

    # 性别 one-hot 编码
    gender_encoder = OneHotEncoder()
    gender_features = torch.tensor(gender_encoder.fit_transform(users[['Gender']]).toarray(), dtype=torch.float)

    # 年龄组嵌入
    num_age_groups = users['Age'].nunique()
    age_group_embedding = nn.Embedding(64, embedding_dim)
    age_group_features = age_group_embedding(torch.tensor(users['Age'].values))

    # # 邮编嵌入
    # num_zipcodes = users['Zip-code'].nunique()
    # zipcode_embedding = nn.Embedding(num_zipcodes, embedding_dim)
    # zipcode_features = zipcode_embedding(torch.tensor(users['Zip-code'].values))

    # 拼接用户特征
    user_features = torch.cat([occupation_features, gender_features, age_group_features], dim=-1)

    return user_features


# 物品特征提取
def extract_item_features(movies, embedding_dim=16, item_emb_size=724442):
    # 电影类型嵌入
    genres = movies['Genres'].str.split('|')
    genre2id = {genre: idx for idx, genre in enumerate(set(sum(genres.tolist(), [])))}
    num_genres = len(genre2id)
    genre_embedding = nn.Embedding(num_genres, embedding_dim)
    genre_features = []
    for movie_genres in genres:
        genre_ids = [genre2id[genre] for genre in movie_genres]
        genre_features.append(genre_embedding(torch.tensor(genre_ids)).mean(dim=0))  # 对多类型取平均
    genre_features = torch.stack(genre_features)

    # # 电影年份嵌入
    # years = movies['Title'].str.extract(r'\((\d{4})\)')[0].fillna('0').astype(int)
    # num_years = years.nunique()
    # year_embedding = nn.Embedding(num_years, embedding_dim)
    # year_features = year_embedding(torch.tensor(years.values))

    # 拼接物品特征
    item_features = genre_features
    return item_features