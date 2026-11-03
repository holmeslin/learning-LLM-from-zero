# 後訓練：RL 與可驗證獎勵

SFT 讓模型照標準答案逐字模仿，但很多任務只能判斷「答得對不對」，不知道標準答案長什麼樣：數學題的最後答案、程式有沒有通過測試。強化學習（reinforcement learning，RL）就是處理這種情況：讓模型自己試，答對的做法加強，答錯的減弱。

這一章先用最小的例子建立直覺：多臂吃角子老虎機上的 policy gradient、REINFORCE，以及 GRPO 用同一個 prompt 的一組回答互相比較的 advantage。接著把它接到語言模型：實作 GRPO 的 clipped loss，用 0／1 的可驗證獎勵（RLVR）訓練模型，並量出正確率的變化。最後看 Dr. GRPO 與 GSPO 改了什麼，以及真實系統為什麼用 vLLM 產生回答。

這一章的程式在 `src/llmz/rl.py`，對應 CS336 對齊（alignment）中 RL 的講次，見 <https://cs336.stanford.edu>。
