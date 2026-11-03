# 後訓練：SFT 與偏好學習

第 5 章訓練出來的模型只會一件事：看到一段文字，接著往下寫。你問它一個問題，它多半不會回答，而是把問題當成故事的開頭繼續編。要讓它「照指令行事」，還需要一段**後訓練**（post-training）。

這一章先看這個差別：拿一份答案可以自動檢查的題目，量 base model 答得出幾題。接著看 chat template 怎麼把一來一往的對話排成一串 token，以及 SFT（supervised fine-tuning，監督式微調）為什麼只在助理的回覆上算 loss。然後在第 5 章的標準模型上做一次 SFT，量出前後的正確率。再來是偏好學習：用成對的「好回答、壞回答」教模型，先看 reward model 與 Bradley–Terry，再從 RLHF 的目標推導出 DPO，自己實作它的 loss。

這一章的程式在 `src/llmz/posttrain.py`，對應 CS336 對齊（alignment）中 SFT 與偏好學習的講次，見 <https://cs336.stanford.edu>。
