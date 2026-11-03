# Scaling laws

訓練一個大模型要花幾個月和大筆經費，不可能先試幾種大小再挑最好的。scaling law（規模定律）讓我們用一批小模型的實驗，預測大模型的 loss，並回答一個很實際的問題：計算預算固定時，模型該多大、該看多少資料？

這一章先認識冪律與 Chinchilla 的結論，再用 Chinchilla 論文的公式產生一組「離線資料」，練習擬合 IsoFLOPs 曲線與冪律。接著在 Mac 上真的訓練 13 個小模型，做一個迷你版的 scaling law。最後談這套方法的限制，以及讓超參數能從小模型搬到大模型的 μP。

這一章對應 CS336 Assignment 3 與 scaling laws 的講次，見 <https://cs336.stanford.edu>。
