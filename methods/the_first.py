import pandas as pd

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

file_path = '../7XdsLqBdlaQ_comments_all.csv'

df = pd.read_csv(file_path)
comment_id_to_f = 167
filtered_comm = df[df['CommentID'] == comment_id_to_f]

print(filtered_comm)