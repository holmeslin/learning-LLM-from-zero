# 生成與實驗

模型訓練好了，這一章先讓它寫故事：從 logits 挑下一個 token 的三種方法（greedy、temperature、top-p），組成完整的生成函式。後半段做實驗：把 Transformer 的零件一個一個拿掉或換掉（RMSNorm、pre-norm、RoPE、SwiGLU），再掃學習率與 batch 大小，用 loss 曲線看每個設計決定值多少。

所有實驗都用縮短的設定（1000 步，約 820 萬個 token），M4 Max 上每個五分鐘左右，加起來一小時；數字都是實測。

這一章對應 CS336 Assignment 1 的生成與消融實驗部分，見 <https://cs336.stanford.edu>。
