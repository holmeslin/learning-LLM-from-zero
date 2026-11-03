# FlashAttention 的原理

第 7 章看到 attention 有個麻煩：分數矩陣的大小跟序列長度的平方成正比，又卡在記憶體頻寬。FlashAttention 用兩個技巧解決它：把計算切成小塊（tiling），以及一邊讀一邊更新的 softmax（online softmax），讓那個 \\( T \times T \\) 的矩陣從頭到尾不用整個存在記憶體裡。backward 時再用「重算」代替「存起來」。

這一章用純 PyTorch 把 FlashAttention 的 forward 與 backward 寫出來，驗證它和第 4 章的 attention 算出一樣的結果與梯度，並量出記憶體從平方降成線性。純 PyTorch 版的 Python 迴圈很慢，真正的速度要靠 GPU kernel；第 6 集簡單看一下 Triton 怎麼寫 kernel，第 7 集看 PyTorch 內建的 `scaled_dot_product_attention` 在各種硬體上現在是什麼狀況。

這一章對應 CS336 Assignment 2 的 FlashAttention 部分，見 <https://cs336.stanford.edu>。
