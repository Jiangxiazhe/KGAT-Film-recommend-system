from neo4j import GraphDatabase
import pandas as pd

# Neo4j 连接配置
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "12345678"

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

# 关系映射表
RELATION_MAP = {
    "0": "RATE",
    "1": "BELONGS_TO",
    "2": "HAS_OCCUPATION",
    "3": "IN_AGE_GROUP",
    "4": "HAS_GENDER",
    "5": "LIVES_IN",
    "6": "RELEASED_IN"
}

# 加载电影数据
movies_df = pd.read_csv('./ml-1m/movies.dat', sep='::', encoding='ISO-8859-1',
                        names=['MovieID', 'Title', 'Genres'])
movie_title_map = {row['MovieID']: row['Title'] for _, row in movies_df.iterrows()}


def load_mappings(entity2id_file):
    """加载实体映射并增强处理"""
    entity_map = {}
    with open(entity2id_file, 'r') as f:
        for line in f:
            entity, idx = line.strip().split('\t')

            # 处理特殊实体类型
            if entity.startswith("Occupation"):
                occ_id = int(entity.replace("Occupation", ""))
                entity = OCCUPATION_MAP.get(occ_id, "unknown_occupation")

            entity_map[idx] = entity
    return entity_map


def get_entity_type(entity_name):
    """获取带语义的实体类型"""
    if entity_name.startswith("User"):
        return ("User", {"user_id": entity_name.replace("User", "")})
    elif entity_name.startswith("Movie"):
        movie_id = entity_name.replace("Movie", "")
        return ("Movie", {
            "movie_id": movie_id,
            "title": movie_title_map.get(int(movie_id), "Unknown")
        })
    elif entity_name.startswith("Gender"):
        return ("Gender", {"gender": entity_name.replace("Gender", "")})
    elif entity_name in OCCUPATION_MAP.values():
        return ("Occupation", {"name": entity_name})
    elif entity_name.startswith("Age_group"):
        return ("AgeGroup", {"range": entity_name.replace("Age_group", "")})
    elif entity_name.startswith("Zipcode"):
        return ("Zipcode", {"code": entity_name.replace("Zipcode", "")})
    elif entity_name.startswith("Year"):
        return ("Year", {"year": entity_name.replace("Year", "")})
    elif entity_name.startswith("Genre"):
        return ("Genre", {"name": entity_name})
    else:
        return ("Entity", {"name": entity_name})


def import_to_neo4j(kg_file, entity2id_file):
    # 加载数据
    entity_map = load_mappings(entity2id_file)

    # 连接 Neo4j
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session() as session:
        # 创建所有节点
        for idx, entity_name in entity_map.items():
            label, properties = get_entity_type(entity_name)

            # 特殊处理电影标题
            if label == "Movie":
                query = """
                MERGE (m:Movie {movie_id: $movie_id})
                SET m.title = $title
                """
                session.run(query,
                            movie_id=properties["movie_id"],
                            title=properties["title"])
            else:
                query = f"MERGE (n:{label} {{ {', '.join([f'{k}: ${k}' for k in properties.keys()])} }})"
                session.run(query, **properties)

        # 创建关系
        with open(kg_file, 'r') as f:
            for line in f:
                head, rel, tail = line.strip().split('\t')

                # 获取关系类型
                rel_type = RELATION_MAP.get(rel, "RELATED_TO")

                # 构建关系
                head_name = entity_map.get(head, "Unknown")
                tail_name = entity_map.get(tail, "Unknown")
                # 获取头实体和尾实体的类型及属性
                head_label, head_properties = get_entity_type(head_name)
                tail_label, tail_properties = get_entity_type(tail_name)

                # 构建关系的 Cypher 查询
                query = f"""
                        MERGE (h:{head_label} {{ {', '.join([f'{k}: ${k}' for k in head_properties.keys()])} }})
                        MERGE (t:{tail_label} {{ {', '.join([f'{k}: ${k}' for k in tail_properties.keys()])} }})
                        MERGE (h)-[:{rel_type}]->(t)
                        """

                # 执行查询
                session.run(query, **head_properties, **tail_properties)

        print("数据导入完成！建议在Neo4j Browser中执行以下查询验证：")
        print("1. 查看用户节点: MATCH (u:User) RETURN u LIMIT 5")
        print("2. 查看电影节点: MATCH (m:Movie) RETURN m LIMIT 5")
        print("3. 查看评分关系: MATCH (u:User)-[r:RATE]->(m:Movie) RETURN u,r,m LIMIT 10")


if __name__ == '__main__':
    import_to_neo4j('kg_final.txt', 'entity2id.txt')