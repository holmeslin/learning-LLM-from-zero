# Transformer 的零件

這一章把語言模型拆成零件，一集做一個：Linear、Embedding、RMSNorm、SwiGLU、RoPE、attention、causal mask、multi-head，最後把它們組成一個 Transformer block，疊起來就是完整的語言模型。每個零件都自己寫，不用 `torch.nn` 的現成層，寫完用 `tests/` 裡的測試和 PyTorch 內建版本對答案。

全章的模型都還沒訓練，輸出是亂數；我們只關心形狀對不對、數學對不對。第 5 章才把資料餵進去。

這一章對應 CS336 Assignment 1 的 Transformer 部分與 Lecture 3，見 <https://cs336.stanford.edu>。
