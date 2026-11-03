# 推論

模型訓練完要拿來用：使用者給一段 prompt，模型一個 token 一個 token 往下寫。訓練只做一次，推論卻要對每個使用者、每個 token 做，所以它的速度與記憶體決定了服務的成本。

這一章先用 KV cache 把第 6 章重複的計算拿掉，再把生成拆成 prefill 與 decode 兩個階段，學會估算延遲與記憶體。接著看三種加速的方法：一次服務很多序列的 batching、用小模型先猜的 speculative decoding，以及把權重存成更少位元的量化。最後一集介紹 MoE 與 attention 的替代方案，它們從架構上降低推論成本。

這一章的程式在 `src/llmz/inference.py`，對應 CS336 推論的講次，見 <https://cs336.stanford.edu>。
