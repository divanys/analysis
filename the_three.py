import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Замените "YOUR_VIDEO_ID_HERE_comments.csv" на путь к вашему CSV-файлу
file_path = "7XdsLqBdlaQ_comments_all.csv"

# Прочитать CSV-файл с помощью pandas
df = pd.read_csv(file_path)

# Получить комментарии в виде списка
comments = df['Comment'].tolist()

# Преобразовать текст комментариев в числовые векторы с помощью TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comments)

# Кластеризовать комментарии методом K-средних
num_clusters = 5  # Укажите желаемое количество кластеров
kmeans = KMeans(n_clusters=num_clusters, random_state=10)
kmeans.fit(X)

# Добавить метки кластеров обратно в DataFrame
df['Cluster'] = kmeans.labels_

# Создать словарь для хранения айди комментариев по кластерам
cluster_dict = {}
for cluster_id in range(num_clusters):
    cluster_dict[cluster_id] = df[df['Cluster'] == cluster_id]['CommentID'].tolist()

# Вывести айди комментариев по кластерам
for cluster_id, comment_ids in cluster_dict.items():
    print(f"Cluster {cluster_id} - Comment IDs: {comment_ids}")