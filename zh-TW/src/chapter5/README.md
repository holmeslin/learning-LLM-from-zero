# 訓練

模型（第 4 章）和資料（第 3 章）都有了，這一章把它們接起來訓練。第 2 章的五步迴圈不變，但每一步裡的東西都要升級：loss 換成 cross-entropy、優化器從 SGD 升級到 AdamW、學習率隨時間變化、梯度要裁剪、資料要隨機取 batch、中途要存 checkpoint、每一步要記 log。

章末用標準語料跑一次完整訓練：4100 萬個 token，M4 Max 上幾十分鐘，得到一個會寫童話的小模型，然後把 loss 曲線畫出來。第 6 章拿這個模型生成文字、做實驗。

這一章對應 CS336 Assignment 1 的訓練部分與 Lecture 3、4，見 <https://cs336.stanford.edu>。
