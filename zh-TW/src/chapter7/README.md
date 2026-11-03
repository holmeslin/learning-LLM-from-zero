# 效能分析與記憶體

第一部的模型在 M4 Max 上訓練了二十幾分鐘。這一章不再加新功能，而是回頭問兩個問題：時間花在哪裡？記憶體被誰吃掉？我們會學正確地計時、用 profiler 找出最慢的運算、把訓練時的記憶體拆成參數、梯度、優化器狀態與 activations 四塊，再試三個常見的加速或省記憶體手段：混合精度、activation checkpointing、`torch.compile`。最後用 arithmetic intensity 把這些現象串起來：一個運算到底是卡在計算還是卡在搬資料。

本章的數字大多在 M4 Max（36 GB）的 MPS 上實測；能在 CPU 上固定重現的部分（例如 activations 的位元組數）用 CPU 示範。

這一章對應 CS336 Assignment 2 的 benchmarking 與 profiling 部分，見 <https://cs336.stanford.edu>。
