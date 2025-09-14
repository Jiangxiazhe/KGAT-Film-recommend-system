import pandas as pd
from torch_geometric.data import Data
import torch

# 加载数据
user_data = pd.read_csv('./kg/user_nodes.csv')  # 用户节点
movie_data = pd.read_csv('./kg/movie_nodes.csv')  # 电影节点
rated_edges = pd.read_csv('./kg/rated_edges.csv')  # 用户评分边
belongs_to_edges = pd.read_csv('./kg/belongs_to_edges.csv')  # 电影类型边

# 提取节点特征
user_features = torch.tensor(user_data[['Age', 'Occupation']].values, dtype=torch.float)
movie_features = torch.tensor(movie_data[['Year']].values, dtype=torch.float)

# 提取边关系
rated_edge_index = torch.tensor(rated_edges[['UserID', 'MovieID']].values, dtype=torch.long).t().contiguous()
belongs_to_edge_index = torch.tensor(belongs_to_edges[['MovieID', 'Genres']].values, dtype=torch.long).t().contiguous()

# 构建图数据
data = Data(
    x_users=user_features,
    x_movies=movie_features,
    edge_index_rated=rated_edge_index,
    edge_index_belongs_to=belongs_to_edge_index
)