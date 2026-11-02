# Summary

[前言](foreword.md)

# 第一部 從零到訓練出一個小 LLM

- [張量與剛好夠用的數學](chapter1/README.md)
    - [安裝 uv 與取得程式碼](chapter1/01_setup.md)
    - [第一個張量](chapter1/02_first_tensor.md)
    - [形狀與 dtype](chapter1/03_shape_dtype.md)
    - [索引與切片](chapter1/04_indexing.md)
    - [broadcasting](chapter1/05_broadcasting.md)
    - [向量與點積](chapter1/06_dot_product.md)
    - [矩陣乘法 `@`](chapter1/07_matmul.md)
    - [`einsum`](chapter1/08_einsum.md)
    - [device：cpu 與 mps](chapter1/09_device.md)
    - [隨機數與種子](chapter1/10_random_seed.md)
    - [導數與鏈鎖律](chapter1/11_derivative_chain_rule.md)
    - [梯度](chapter1/12_gradient.md)
    - [softmax](chapter1/13_softmax.md)
    - [機率、log 機率與取樣](chapter1/14_probability_sampling.md)
    - [cross-entropy 與 perplexity](chapter1/15_cross_entropy_perplexity.md)
    - [數值穩定：log-sum-exp](chapter1/16_logsumexp.md)
- [autograd 與訓練迴圈]()
- [Tokenizer：從 bytes 到 token]()
- [Transformer 的零件]()
- [訓練]()
- [生成與實驗]()
- [附錄一]()

# 第二部 系統、規模與推論

- [效能分析與記憶體]()
- [FlashAttention 的原理]()
- [平行化：原理與 CPU 模擬]()
- [Scaling laws]()
- [推論]()
- [附錄二]()

# 第三部 評估、資料與對齊

- [評估]()
- [資料：從網頁到訓練集]()
- [後訓練：SFT 與偏好學習]()
- [後訓練：RL 與可驗證獎勵]()
