分析コンペ ➡️ データ分析コンペティション

データ理解(EDA)：探査的データ分析(Exploratory Data Analysis, EDA)とも呼ばれます。モデルや特徴量を作る上でまず優先すべきなのがEDAです。データの性質を応じて、さまざまな手法を用いられます。

統計量：

* 変数の平均、標準偏差、最大、最小、分位点；
* カテゴリ変数の値の種類数
* 変数の欠損値
* 変数間の相関係数

可視化手法：

+ 棒グラフ
+ 箱ヒゲ図、バイオリンプロット
+ 散布図
+ 折れ線グラフ
+ ヒートマップ
+ ヒストグラム
+ Q-Qプロット
+ t-SNE、UMAP

分類コンペにおけるタスクの種類：回帰タスク、分類タスク、レコメデーション、物体検出(Object Detection)、セグメンテーション(Segmentation)。

評価指標(Evaluation metrics)：

+ 回帰における評価指標：RMSE(平均平方二乗誤差)、RMSLE(Root Mean Squared Logarithmic Error)、MAE(Mean Absolute Error)、決定係数。
+ 二値分類における評価指標：混同行列(Confusion Matrix)、Accuray(正答率)/ErrorRate(誤答率)、Precision(適合率)/Recall(再現率)、F1-score/F$\beta$-score、MCC
+ logloss、AUC(Area Under the ROC Curve)、
+ 多クラス分類における評価指標：multi-class accurary、multi-class accurary、mean-F1/macro-F1/micro-F1、Quadratic Weighted Kappa
+ レコメンデーションにおける評価指標：MAP@K(Mean Average Precision)

評価指標の最適化方法

## 2.7 リーク(Data leakage)





