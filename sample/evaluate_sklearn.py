# coding:utf-8
# モデルの評価
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib # 追加
matplotlib.use('TkAgg') # 追加
import matplotlib.pyplot as plt

N = 300
# 月形のデータを生成。0.3 はデータがどのくらい重なり合うか
X, y = datasets.make_moons(N, noise=0.3)
# 描画で確認
# 散布図を描画
for i in range(N):
    if(y[i] == 0):
        plt.scatter(X[i][0], X[i][1], c='red')
    else:
        plt.scatter(X[i][0], X[i][1], c='blue')
plt.show()

# データの次元を合わせる
Y = y.reshape(N, 1)
# 訓練データとテストデータを8:2に分ける
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, train_size=0.8)

# モデル構築
num_hidden = 2
# placeholder: 定義の時は次元だけでいい None, 2 は2次元のが？個ということ
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
# 入力 - 隠れ層
W = tf.Variable(tf.truncated_normal([2, num_hidden]))
b = tf.Variable(tf.zeros([num_hidden]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)
# 隠れ層 - 出力層
W = tf.Variable(tf.truncated_normal([num_hidden, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(x, W) + b)
