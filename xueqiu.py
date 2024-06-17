import requests
import json
import jieba
import jieba.analyse as anls
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud


# 爬取博客园的热门新闻
def get_words():
  base_url = 'https://xueqiu.com/v4/statuses/user_timeline.json?page='
  header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'cookie': 'xq_a_token=9340aab2bf1fc4365f85394da9debf5962df2db7'
  }
  for i in range(100):
    url = base_url + str(i + 1) + '&user_id=5124430882'
    response = requests.get(url, headers=header)
    ret = json.loads(response.text)
    list = ret['statuses']
    with open('xueqiu/words.txt', 'a', encoding='utf-8') as fp:
      for item in list:
        print(item['text'])
        fp.write(item['text'] + '\n')


# 从热门新闻中过滤出信息化的词语
def words_filter():
  jieba.load_userdict('data/dict.txt')
  jieba.analyse.set_stop_words('data/stop_words.txt')
  text = ''
  with open('xueqiu/words.txt', encoding='utf-8') as fp:
    text = fp.read()
  words_list = jieba.lcut(text, False)
  words_filter_list = [
    word
    for word in words_list
    if word in jieba.dt.FREQ and len(word) > 1 and not word.isnumeric()
  ]
  with open('xueqiu/words_res.txt', 'a', encoding='utf-8') as fp:
    for word in words_filter_list:
      fp.write(word + '\n')
  res = ''
  with open('xueqiu/words_res.txt', encoding='utf-8') as fp:
    res = fp.read()
  print('基于TF-IDF提取关键词结果：')
  with open('xueqiu/words_ans.txt', 'a', encoding='utf-8') as fp:
    for x, w in anls.extract_tags(res, topK=100000, withWeight=True):
      fp.write(f'{x}\t{w}\n')
      print(f'{x}\t{w}')


# 生成词云
def word_cloud():
  word_list = []
  with open('xueqiu/words_ans.txt', encoding='utf-8') as fp:
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
  # plt.show()
  wordcloud.to_file('xueqiu/word_cloud.jpg')


if __name__ == '__main__':
  if os.path.exists('xueqiu/words.txt'):
    os.remove('xueqiu/words.txt')
  if os.path.exists('xueqiu/words_res.txt'):
    os.remove('xueqiu/words_res.txt')
  if os.path.exists('xueqiu/words_ans.txt'):
    os.remove('xueqiu/words_ans.txt')
  get_words()
  words_filter()
  word_cloud()
