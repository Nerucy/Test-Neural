# coding:utf-8
# 生成したデータをパーセプトロンで分類してみる

import matplotlib # 追加
matplotlib.use('TkAgg') # 追加

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(123)

d = 2  # データの次元
N = 10  # 各パターンのデータ数
mean = 5  # ニューロンが発火するデータの平均数

# randn(N行, d列) の標準正規分布の乱数
x_0 = rng.randn(N, d) + np.array([0, 0])
x_1 = rng.randn(N, d) + np.array([mean, mean])
x = np.concatenate((x_0, x_1), axis=0)

# 重みベクトルとバイアスbを初期化
w = np.zeros(d)
b = 0

# 出力の一般式は y = f(wt x + b)
def y(x):
    return step(np.dot(w, x) + b)

def step(xx):
    return 1 * (xx > 0)  # ステップ関数

# 正解出力値を定義
def t(i):
    if i < N:  # 前半10個はx1のデータ
        return 0
    else:
        return 1

# 誤り訂正学習法
while True:
    classified = True
    for i in range(N * 2):
        delta_w = (t(i) - y(x[i])) * x[i]
        delta_b = (t(i) - y(x[i]))
        w += delta_w
        b += delta_b
        classified *= all(delta_w == 0) * (delta_b == 0)
    if classified:
        break

    print("w =", w)
    print("b =", b)

# 描画で確認
# 散布図を描画
for i in range(N):
    plt.scatter(x_0[i][0], x_0[i][1], c='red')
    plt.scatter(x_1[i][0], x_1[i][1], c='blue')

# numpy内のarangeメソッドを用いて値の変域と刻みを設定
# y = f(x1 * w[0] + x2 * w[1] + b)
xx = np.arange(-3, 7, 0.1)
yy = -1 * xx * w[0]/w[1] - b

plt.plot(xx, yy)

plt.show()
