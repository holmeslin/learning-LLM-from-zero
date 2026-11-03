# 消融：post-norm

## 本集目標

把 RMSNorm 從子層之前移到殘差相加之後（post-norm），看第 4 章第 13 集說的「pre-norm 比較穩」。

## 正文

### 差在哪裡

```text
pre-norm（本書預設）            post-norm
x = x + attn(norm(x))          x = norm(x + attn(x))
x = x + ffn(norm(x))           x = norm(x + ffn(x))
```

post-norm 是最早的 Transformer 用法。差別在殘差那條路：pre-norm 從第一層到最後一層是純加法，梯度直通；post-norm 每一層都把整個 `x` 重新正規化，梯度每過一層就被 norm 的導數改一次，層數多了訊號會變形。

```bash
uv run tools/train.py --out_dir runs/post_norm --compile --max_steps 1000 --eval_every 100 --norm_position post
```

### 結果

| run | 第 200 步驗證 loss | 第 1000 步驗證 loss | 平均梯度長度 |
| --- | --- | --- | --- |
| pre-norm（基準） | 3.08 | 2.17 | 0.52 |
| post-norm | 3.03 | **2.14** | 0.60 |

![pre-norm 與 post-norm](images/ablation_post_norm.png)

post-norm 反而低了 0.02，在第 5 集說的雜訊範圍內，算平手。梯度長度也正常，沒有任何不穩的跡象。

這不代表書上第 4 章說錯了，而是**實驗的尺寸決定了你看得到什麼**。post-norm 的問題是梯度每過一層就被 norm 改一次，4 層只改 4 次，影響很小；幾十層疊起來才會累積成問題，而且多半出現在大學習率、短 warmup 的時候。本書的 4 層模型加上 200 步 warmup 剛好落在安全區。

這是消融實驗最重要的一課：小模型上看不到差別，不能推論到大模型也沒差別。想親眼看到 post-norm 出問題，可以試試 `--n_layers 16 --warmup_steps 20`（會慢很多），自己比較。

## 重點整理

- post-norm：`norm(x + sublayer(x))`；pre-norm：`x + sublayer(norm(x))`。
- 4 層模型上 post-norm 與 pre-norm 平手（2.14 對 2.17），沒有不穩。
- 差別要在層數多、warmup 短時才顯現；小模型的消融結果不能直接推到大模型。
- 現代 LLM 幾乎都用 pre-norm，原因就是這條曲線。
