import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 确保你已经下载了nltk的必要数据
nltk.download('punkt')
nltk.download('stopwords')

# 示例英文文章
text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence
concerned with the interactions between computers and human language, in particular how to program computers to
process and analyze large amounts of natural language data.
"""

# 分词
words = word_tokenize(text)

# 将词语转换为小写
words = [word.lower() for word in words]

# 移除标点符号
words = [word for word in words if word not in string.punctuation]

# 移除停用词
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# 统计词频
word_counts = Counter(words)

# 生成词云
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
