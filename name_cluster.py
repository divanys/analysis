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

# Создать словарь для хранения тематических названий кластеров
cluster_topics = {}
for cluster_id in range(num_clusters):
    cluster_comments = df[df['Cluster'] == cluster_id]['Comment']
    # Используем наиболее часто встречающееся слово из комментариев как тематическое название кластера
    top_word = cluster_comments.str.split().explode().mode().values[0]
    cluster_topics[cluster_id] = top_word

# Вывести тематические названия кластеров
for cluster_id, topic in cluster_topics.items():
    print(f"Cluster {cluster_id} - Topic: {topic}")