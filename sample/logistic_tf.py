# coding:utf-8
# ロジスティック回帰でOR ゲート(入力は2次元, 出力は1次元)
import numpy as np
import tensorflow as tf

# w = 重み　b = バイアス
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
print(w)
print(b)

# モデル構築。yが出力。シグモイド関数
# y = σ(wt x + b)
# placeholde: 定義の時は次元だけでいい None, 2 は2次元のが？個ということ
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1]) # 正解値
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

# 交差エントロピー誤差関数
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
# 各パラメータでの偏微分 -> 確率的勾配降下法 今回は学習率0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# y >= 0.5 なら発火
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 学習用データ準備
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# おきまりらしい
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 学習
# placeholder にfeed_dict で代入
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

# 学習データでできてるか確認
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
print(classified)

# 各入力に対する出力確率
prob = y.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
print(prob)

# 変数の確認
print('w: ', sess.run(w))
print('b: ', sess.run(b))
