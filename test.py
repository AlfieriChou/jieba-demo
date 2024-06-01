import requests
from bs4 import BeautifulSoup
import jieba
import jieba.analyse as anls
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud


# 爬取博客园的热门新闻
def get_words():
  url = 'https://news.cnblogs.com/n/recommend'
  header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188'
  }
  for i in range(100):
    parm = {'page': i + 1}
    response = requests.request(
      method='post', url=url, params=parm, headers=header
    )
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'lxml')
    h2 = soup.find_all('h2', class_='news_entry')
    with open('data/words.txt', 'a', encoding='utf-8') as fp:
      for j in h2:
        fp.write(j.find('a').text + '\n')


# 从热门新闻中过滤出信息化的词语
def words_filter():
  jieba.load_userdict('data/dict.txt')
  jieba.analyse.set_stop_words('data/stop_words.txt')
  text = ''
  with open('data/words.txt', encoding='utf-8') as fp:
    text = fp.read()
  words_list = jieba.lcut(text, False)
  words_filter_list = [
    word
    for word in words_list
    if word in jieba.dt.FREQ and len(word) > 1 and not word.isnumeric()
  ]
  with open('data/words_res.txt', 'a', encoding='utf-8') as fp:
    for word in words_filter_list:
      fp.write(word + '\n')
  res = ''
  with open('data/words_res.txt', encoding='utf-8') as fp:
    res = fp.read()
  print('基于TF-IDF提取关键词结果：')
  with open('data/words_ans.txt', 'a', encoding='utf-8') as fp:
    for x, w in anls.extract_tags(res, topK=10000, withWeight=True):
      fp.write(f'{x}\t{w}\n')
      print(f'{x}\t{w}')


# 生成词云
def word_cloud():
  word_list = []
  with open('data/words_ans.txt', encoding='utf-8') as fp:
    line = fp.readline()
    while line:
      word_line = line.strip().split('\t')
      word_list.append(word_line[0])
      line = fp.readline()
  text = '/'.join(word_list)
  maskph = np.array(Image.open('data/LuXun_black.jpg'))
  wordcloud: WordCloud = WordCloud(
    mask=maskph,
    background_color='white',
    font_path='data/SimHei.ttf',
    margin=2,
  ).generate(text)

  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show()
  wordcloud.to_file('data/word_cloud.jpg')


if __name__ == '__main__':
  if os.path.exists('data/words.txt'):
    os.remove('data/words.txt')
  if os.path.exists('data/words_res.txt'):
    os.remove('data/words_res.txt')
  if os.path.exists('data/words_ans.txt'):
    os.remove('data/words_ans.txt')
  get_words()
  words_filter()
  word_cloud()
