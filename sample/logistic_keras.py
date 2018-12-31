# coding:utf-8
import numpy as np
# 層構造のモデルを定義するためのメソッド
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

model = Sequential([
    # 入力: 2次元、出力: 1次元のネットワーク構造を持つ層
    Dense(input_dim=2, units=1), # (w1x1 + w2x2 + b)
    Activation('sigmoid')        # y = σ(w1x1 + w2x2 + b)
])

# 書き方はこっちでも OK
# model = Sequential()
# model.add(Dense(input_dim=2, units=1))
# model.add(Activation('sigmoid'))

# 確率的勾配降下法
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

# 正解値
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# 実行これだけ
model.fit(X, Y, epochs=200, batch_size=1)

# 確認
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)