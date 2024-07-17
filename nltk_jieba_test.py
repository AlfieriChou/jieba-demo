import nltk
import jieba
import jieba.analyse as anls
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import re

# 确保你已经下载了nltk的必要数据
nltk.download('punkt')
nltk.download('stopwords')

# 设置matplotlib后台执行
matplotlib.use('agg')

# 示例中英文文章
text = """
自然语言处理（Natural Language Processing，NLP）是计算机科学、人工智能和语言学领域的一个重要方向。
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，所以它与语言学的研究有着密切的联系，但又有重要的区别。
自然语言处理并不是一般地研究自然语言，而在于研制能有效地实现自然语言通信的计算机系统，特别是其中的软件系统。
"""

# 分离英文和中文部分（简单处理）
english_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
chinese_text = re.sub(r'[^\u4e00-\u9fa5]+', ' ', text)


# 英文分词
words_en = word_tokenize(english_text)

# 初始化jieba
jieba.load_userdict('data/dict.txt')
anls.set_stop_words('data/stop_words.txt')

# 中文分词
words_zh = jieba.lcut(chinese_text, False)
words_zh = [
  word
  for word in words_zh
  if word in jieba.dt.FREQ and len(word) > 1 and not word.isnumeric()
]

# 合并中英文分词结果
words = words_en + words_zh

# 将词语转换为小写（仅适用于英文单词）
words = [word.lower() for word in words]

# 移除标点符号
words = [word for word in words if word not in string.punctuation]

# 移除停用词（仅适用于英文单词）
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# 统计词频
word_counts = Counter(words)

# 图片背景
maskph = np.array(Image.open('data/LuXun_black.jpg'))

# 生成词云
wordcloud = WordCloud(
  mask=maskph,
  background_color='white',
  font_path='data/SimHei.ttf',
  margin=2,
).generate('/'.join(word_counts))

# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.show()

# 保存词云
wordcloud.to_file('data/nltk_jieba_word_cloud.jpg')
