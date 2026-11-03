# 為什麼需要多卡

## 本集目標

用第 2 章與第 7 章的估算方法，算出大模型為什麼一張卡放不下、也算不完，並認識四種把工作分出去的方法。

## 正文

### 放不下

第 7 章第 4 集算過，用 AdamW 訓練每個參數要 16 位元組（參數、梯度、兩個優化器狀態），還沒算 activations。換成一個 70 億參數的模型：

```python
n_params = 7e9
print(f"{16 * n_params / 1e9:.0f} GB")
```

執行結果：

```text
112 GB
```

一張 NVIDIA H100 有 80 GB 記憶體。光是參數、梯度、優化器狀態就放不下，更別說 activations。

### 算不完

第 2 章第 7 集的估算：訓練一個 token 約 \\( 6 \times \\) 參數量個 FLOPs。假設這個模型看 1 兆個 token，一張 H100 的 bfloat16 規格是約 990 TFLOPs（第 7 章第 1 集），實際訓練能用到的比例我們假設是 40%：

```python
n_params = 7e9
tokens = 1e12
flops = 6 * n_params * tokens
seconds = flops / (0.4 * 990e12)
print(f"{flops:.1e} FLOPs")
print(f"{seconds / 86400:.0f} 天")
```

執行結果：

```text
4.2e+22 FLOPs
1228 天
```

一張卡要算三年多。用 1000 張卡，如果能完美分工，大約一天。實際上卡之間要通訊，不可能完美，這一章的重點就是怎麼分工、要付出多少通訊。

### 四種分法

| 方法 | 每張卡負責 | 每步要通訊什麼 | 本章 |
| --- | --- | --- | --- |
| data parallel | 不同的資料，完整的模型 | 梯度（all-reduce） | 第 4、5 集 |
| sharding（ZeRO、FSDP） | 不同的資料，但優化器狀態、梯度、參數各存一部分 | 梯度與參數的片段 | 第 6、7 集 |
| tensor parallel | 同一層的一部分 | 每一層的 activations | 第 8 集 |
| pipeline parallel | 不同的層 | 層與層之間的 activations | 第 8 集 |

實際的大模型訓練通常同時用好幾種：同一台機器內的 8 張卡做 tensor parallel（卡之間的連線最快），機器之間做 pipeline 與 data parallel。

### 這一章怎麼模擬

我們沒有多張 GPU，但多卡訓練的程式其實是**很多個 process 各自執行同一份程式**，每個 process 控制一張卡，彼此透過通訊函式交換資料。PyTorch 的 `torch.distributed` 在 NVIDIA 卡上用 NCCL 通訊，在 CPU 上用 Gloo。我們用 Gloo 在一台 Mac 上起 2 到 4 個 process，每個假裝是一張卡；通訊的寫法和真正的多卡訓練一樣，只是速度不同。

## 重點整理

- 70 億參數的模型用 AdamW 訓練，光參數、梯度、優化器狀態就要 112 GB，一張 80 GB 的卡放不下。
- 同一個模型看 1 兆個 token 要 \\( 4.2 \times 10^{22} \\) FLOPs，一張 H100 要算三年多。
- 四種分法：data parallel、sharding、tensor parallel、pipeline parallel，實務上混著用。
- 多卡訓練是好幾個 process 各跑同一份程式，靠通訊交換資料；本章用 Gloo 在 CPU 上模擬。
