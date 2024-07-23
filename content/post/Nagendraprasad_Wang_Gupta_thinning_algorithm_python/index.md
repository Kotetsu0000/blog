+++
title = '【Python】NWG細線化を実装してみた'
date = 2024-07-07T03:46:53+09:00
description = "細線化アルゴリズムはいくつかあり、「Zhang-Suen法」や「田村法」等ある。「Zhang-Suen法」についてはPythonでの実装例が見つかったが、NWG法は実装例がなかったため、実装することにした。"
categories = ["Python", "Image Processing"]
tags = ["Python", "Image Processing"]
+++

細線化アルゴリズムはいくつかあり、「Zhang-Suen法」や「田村法」等ある。「Zhang-Suen法」についてはPythonでの実装例が見つかったが、NWG法は実装例がなかったため、実装することにした。(もしかしたらどこかのライブラリで既に実装されているかもしれないが)

## NWG細線化とは？

NWG法とは Nagendraprasad-Wang-Guptaの細線化アルゴリズムのことで、細線化アルゴリズムの一つである。(参考: [A note on the Nagendraprasad-Wang-Gupta thinning algorithm](https://www.sciencedirect.com/science/article/pii/016786559500121V))

## アルゴリズム1

**入力:** 
* 二値画像Q (1は黒画素、0は白画素を表す)

**出力:** 
* 細線化された二値画像Q

**処理手順:**

1. 変数g, hを1で初期化する。二値画像QのコピーをQ'とする。
2. hが1である間、以下の処理を繰り返す。
    * hを0にする。
    * gの値を反転させる (1 -> 0, 0 -> 1)。
    * Q'の内容をQにコピーする。
    * Q内の全ての画素pについて、以下の処理を行う。
        * pの8近傍の画素値に基づいて、b(p), a(p), c(p), e(p), f(p)を計算する (詳細は後述)。
        * もし1 < b(p) < 7 かつ (a(p) = 1 もしくは c(p) = 1) ならば、以下の処理を行う。
            * g = 0 かつ e(p) = 0 ならば、Q'内の画素pを白画素(0)にする。
            * g = 1 かつ f(p) = 0 ならば、Q'内の画素pを白画素(0)にする。
3. 細線化処理が完了したので、Qを返す。

**各変数の意味:**

* **p(n):** 画素pの8近傍の画素を表す (n = 0, 1, ..., 7)。
    * 例えば、p(0)はpの右隣の画素、p(1)はpの右斜め下の画素を表す。
* **b(p):** pの8近傍にある黒画素の数。
* **a(p):** pの8近傍を時計回りもしくは反時計回りに一周するとき、白画素から黒画素への遷移回数。
* **c(p), d(p):** pとその8近傍の画素の配置パターンに基づいて、1または0を取る。特定のパターンにのみ1となる。
* **e(p), f(p):** pとその8近傍の画素の配置パターンに基づいて、1または0を取る。特定のパターンにのみ1となる。

## アルゴリズム2

アルゴリズム1は非対称な処理のため、本来削除されるべきでない画素まで削除してしまう場合がある。そこで、アルゴリズム1の非対称性を解消するために、以下の変更を加えたアルゴリズムを**対称型NWGアルゴリズム**と呼ぶ。

* アルゴリズム1の 7. の条件式 `(a(p) = 1 or c(p) = 1)` を `(a(p) = 1 or (1-g) * c(p) + g * d(p) = 1)` に変更する。
* d(p)はc(p)と同様に、pとその8近傍の画素の配置パターンに基づいて、1または0を取る。特定のパターンにのみ1となる。

この変更により、より正確な細線化結果を得ることができる。

## PythonによるNWG法の実装

以上を踏まえ、NWG細線化の実装を行なった。Pythonで実装する際には、Numpyを使用しないと遅いのでOpenCV+Numpyを前提とした実装を行なった。

```python
def NWG(img:np.ndarray, symmetric:bool=False) -> np.ndarray:
    src = np.copy(img)//255

    # zero padding
    ROW, COLUMN = src.shape[0]+2, src.shape[1]+2
    pad = np.zeros((ROW, COLUMN))
    pad[1:ROW-1, 1:COLUMN-1] = src
    src = pad

    switch = True
    while True:
        r, c = src.nonzero()
        nei = np.array((src[r-1, c], src[r-1, c+1], src[r, c+1],
                        src[r+1, c+1], src[r+1, c], src[r+1, c-1],
                        src[r, c-1], src[r-1, c-1]))

        # condition 1
        nei_sum = np.sum(nei, axis=0)
        cond1 = np.logical_and(2 <= nei_sum, nei_sum <= 6)

        # condition 2
        nei = np.concatenate([nei, np.array([nei[0]])], axis=0)
        cond2 = np.zeros(nei.shape[1], dtype=np.uint8)
        for i in range(1, 9):
            cond2 += np.array(
                np.logical_and(nei[i-1] == 0, nei[i] == 1), dtype=np.uint8)
        cond2 = cond2 == 1
        nei = nei[0:8]

        # condition 3
        if symmetric:
            if switch:
                c3a = np.logical_and(
                    nei[0]+nei[1]+nei[2]+nei[5] == 0, nei[4]+nei[6] == 2)
                c3b = np.logical_and(
                    nei[2]+nei[3]+nei[4]+nei[7] == 0, nei[0]+nei[6] == 2)
                cond3 = np.logical_or(c3a, c3b)
            else:
                c3c = np.logical_and(
                    nei[1]+nei[4]+nei[5]+nei[6] == 0, nei[0]+nei[2] == 2)
                c3d = np.logical_and(
                    nei[0]+nei[3]+nei[6]+nei[7] == 0, nei[2]+nei[4] == 2)
                cond3 = np.logical_or(c3c, c3d)
        else:
            c3a = np.logical_and(
                nei[0]+nei[1]+nei[2]+nei[5] == 0, nei[4]+nei[6] == 2)
            c3b = np.logical_and(
                nei[2]+nei[3]+nei[4]+nei[7] == 0, nei[0]+nei[6] == 2)
            cond3 = np.logical_or(c3a, c3b)    

        # condition 4
        if switch:
            cond4 = (nei[2]+nei[4])*nei[0]*nei[6] == 0
        else:
            cond4 = (nei[0]+nei[6])*nei[2]*nei[4] == 0

        cond = np.logical_and(cond1, np.logical_or(cond2, cond3))
        cond = np.logical_and(cond, cond4)
        if True in cond:
            switch = not switch
        else:
            return (src[1:ROW-1, 1:COLUMN-1]*255).astype(np.uint8)

        src[r[cond], c[cond]] = 0
```

## 次回

Cythonを利用した高速化について書く予定...
