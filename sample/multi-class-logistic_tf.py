# coding:utf-8
# 多クラスロジスティック回帰(入力は2次元, 出力は3次元), ミニバッチ
import numpy as np
import tensorflow as tf

import matplotlib # 追加
matplotlib.use('TkAgg') # 追加
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

M = 2     # 入力は2次元
K = 3     # 出力クラス数
n = 100   # クラスごとのデータ数
N = n * K # 全データ数

# サンプルデータ
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

# w = 重み　b = バイアス
w = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

# モデル構築。yが出力。softmax関数
# y = softmax(wt x + b)
# placeholder: 定義の時は次元だけでいい None, 2 は2次元のが？個ということ
x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K]) # 正解値
y = tf.nn.softmax(tf.matmul(x, w) + b)

# ミニバッチごとの平均値
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y),
                                              reduction_indices=[1]))

# 確率的勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# argmaxの第二引数=行ごとという意味
# y は確率。softmax の特徴で1要素のみ値が大きくなる
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

# 学習
batch_size = 50
n_batches = N

# おきまりらしい
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 50個ずつずらして見ていく
for epoch in range(20):
    X_, Y_ = shuffle(X, Y)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        # feed_dict でx, t に代入
        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })

# 確認
X_, Y_ = shuffle(X, Y)
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print('classified:')
print(classified)
print('output probability:')
print(prob)

# 描画で確認
# 散布図を描画
for i in range(n):
    plt.scatter(X1[i][0], X1[i][1], c='red')
    plt.scatter(X2[i][0], X2[i][1], c='purple')
    plt.scatter(X3[i][0], X3[i][1], c='blue')


# 境界はsoftmax の中身が等しくなるところ
xx = np.arange(-2, 10, 0.1) # numpy内のarangeメソッドを用いて値の変域と刻みを設定
W = sess.run(w) #W: 入力行、出力列
B = sess.run(b)
print(W[1][2])
print(B)
# w[0][0]x[0]+w[1][0]x[1]+b[0] = w[0][1]x[0]+w[1][1]x[1]+b[1]
y1 = (W[0][1]*xx - W[0][0]*xx - B[0] + B[1]) / (W[1][0] - W[1][1])
# w[0][1]x[0]+w[1][1]x[1]+b[1] = w[0][2]x[0]+w[1][2]x[1]+b[2]
y2 = (W[0][2]*xx - W[0][1]*xx - B[1] + B[2]) / (W[1][1] - W[1][2])

plt.plot(xx, y1)
plt.plot(xx, y2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()