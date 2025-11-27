import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载模型
with open('movie_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 加载数据
df = pd.read_csv('processed_data.csv')
user_movie_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')
user_movie_matrix.fillna(0, inplace=True)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
def recommend_for_user(user_id, top_n=10):
    # 找到最相似的用户
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:11]  # 排除自己
    
    # 获取目标用户未评分的电影
    user_rated_movies = set(df[df['user_id'] == user_id]['movie_id'])
    all_movies = set(df['movie_id'])
    unrated_movies = all_movies - user_rated_movies
    
    # 预测评分：使用相似用户的评分加权平均
    movie_scores = {}
    for movie in unrated_movies:
        weighted_sum = 0
        sim_sum = 0
        for sim_user in similar_users:
            sim = user_similarity_df.loc[user_id, sim_user]
            rating = user_movie_matrix.loc[sim_user, movie]
            if rating > 0:
                weighted_sum += sim * rating
                sim_sum += sim
        if sim_sum > 0:
            movie_scores[movie] = weighted_sum / sim_sum
    
    # 返回评分最高的电影
    recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommended_movies, similar_users
if __name__ == "__main__":
    user_id = int(input("请输入用户ID: "))
    
    if user_id not in df['user_id'].values:
        print("用户ID不存在")
    else:
        recommendations, similar_users = recommend_for_user(user_id)
        
        print(f"\n与用户 {user_id} 最相似的其他用户：")
        for i, sim_user in enumerate(similar_users, 1):
            print(f"{i}. 用户 {sim_user}")
        
        print(f"\n为用户 {user_id} 推荐的电影(电影ID - 预测评分）：")
        for i, (movie_id, score) in enumerate(recommendations, 1):
            movie_title = df[df['movie_id'] == movie_id]['title'].iloc[0]
            print(f"{i}. {movie_title} (ID: {movie_id}) - 预测评分: {score:.2f}")
