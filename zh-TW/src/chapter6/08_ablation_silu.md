# 消融：SiLU 取代 SwiGLU

## 本集目標

把 SwiGLU 的 gate 拿掉，換成最簡單的 `w2(silu(w1(x)))`，在參數量相同的條件下比較。

## 正文

### 公平比較

SwiGLU 三個矩陣 \\( 3 \times d \times d_{\text{ff}} \\)，標準設定 \\( d_{\text{ff}} = 1344 \\)，每層 206 萬個參數。沒有 gate 的 FFN 只有兩個矩陣，如果 `d_ff` 不變，參數少三分之一，輸了也不知道是少了 gate 還是少了參數。所以把 `d_ff` 加到 2048：\\( 2 \times 512 \times 2048 = 210 \\) 萬，和 SwiGLU 幾乎一樣（第 4 章第 5 集說的 \\( 4d \\) 慣例就是這個）。

```bash
uv run tools/train.py --out_dir runs/silu --compile --max_steps 1000 --eval_every 100 --ffn silu --d_ff 2048
```

### 結果

| run | FFN 參數（每層） | 第 200 步驗證 loss | 第 1000 步驗證 loss |
| --- | --- | --- | --- |
| SwiGLU，`d_ff` 1344（基準） | 2,064,384 | 3.08 | **2.17** |
| SiLU，`d_ff` 2048 | 2,097,152 | 3.19 | 2.27 |

![SwiGLU 與 SiLU](images/ablation_silu.png)

參數量幾乎相同（SiLU 版還多 1.6%），SwiGLU 低了 0.10，超過雜訊的範圍。gate 讓模型能逐維度決定「這一維要放多少資訊通過」，同樣的參數用得更有效率。這也是為什麼現在的 LLM 幾乎都用 GLU 家族的 FFN。

## 重點整理

- 比較不同結構時先把參數量對齊：SiLU FFN 用 \\( d_{\text{ff}} = 4d \\)，SwiGLU 用 \\( 8d/3 \\)。
- 參數量對齊後，SwiGLU 比沒有 gate 的 SiLU FFN 低 0.10（2.17 對 2.27）。
