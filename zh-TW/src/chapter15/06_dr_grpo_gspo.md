# Dr. GRPO、GSPO 一瞥

## 本集目標

知道 GRPO 裡「除以標準差」與「除以回答長度」這兩個除法，Dr. GRPO 為什麼主張拿掉；知道 GSPO 把 token 層級的比值換成整條回答一個比值的動機。用 `grpo_loss` 的參數切換這幾種做法，量出對 RLVR 正確率的影響。

## 正文

### 回頭看 GRPO 的兩個除法

前兩集留下兩個沒回答的問題：第 3 集的 `group_advantages` 要不要除以組內標準差，第 4 集的 `grpo_loss` 要不要除以回答的 token 數（`length_norm`）。把原版 GRPO 的 loss 寫出來，兩個除法都在裡面（\\( r_i \\) 是第 \\( i \\) 條回答的獎勵）：

\\[ A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}, \qquad \mathcal{L} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\big( \rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1\pm\varepsilon) A_i \big) \\]

Liu 等人（2025，arXiv 2503.20783）分析 R1-Zero 式的訓練，指出這兩個除法各帶來一種偏差，提出 Dr. GRPO（GRPO Done Right，意思是「把 GRPO 做對」）：把它們拿掉。

**長度偏差（除以 \\( |o_i| \\)）**。每條回答的總權重一樣，所以回答越長，每個 token 分到的越少。advantage 為正（答對）時，短的回答每個 token 被推得比較用力，模型偏好簡短的正確答案；advantage 為負（答錯）時，長的回答每個 token 被壓得比較輕，模型寧願把錯的答案寫長。用 `grpo_loss` 的梯度看：兩條答錯的回答（advantage 都是 -1），一條 2 個 token、一條 8 個 token：

```python
import torch

from llmz.rl import grpo_loss

mask = torch.tensor([[True] * 2 + [False] * 6, [True] * 8])
for name, kwargs in [("length_norm=True（GRPO）", dict(length_norm=True)), ("固定除數 8（Dr. GRPO）", dict(length_norm=False, length_divisor=8.0))]:
    logp = torch.zeros(2, 8, requires_grad=True)
    grpo_loss(logp, torch.zeros(2, 8), -torch.ones(2), mask, **kwargs).backward()
    short, long = logp.grad[0, 0].item(), logp.grad[1, 0].item()  # 梯度是正的：梯度下降會把 logp 往下壓
    print(f"{name}：短答案每個 token {short:.4f}（共 {2 * short:.3f}），長答案每個 token {long:.4f}（共 {8 * long:.3f}）")
```

執行結果：

```text
length_norm=True（GRPO）：短答案每個 token 0.2500（共 0.500），長答案每個 token 0.0625（共 0.500）
固定除數 8（Dr. GRPO）：短答案每個 token 0.0625（共 0.125），長答案每個 token 0.0625（共 0.500）
```

GRPO 對兩條回答各壓 0.5，長的那條分攤到 8 個 token，每個只剩 0.0625，是短答案的四分之一：答錯的時候，寫長一點比較便宜。Dr. GRPO 改成除以同一個固定的數（論文建議用最大生成長度），每個 token 分到的一樣，長的那條總共被壓得多（0.5 對 0.125）。我們的 `length_norm=False` 搭配 `length_divisor`（這裡取 `max_new_tokens`）就是這個做法。這個除數只是一個常數：各個 token 之間的相對權重不變，只有整個 loss 的大小變。`AdamW` 的更新量對梯度整體放大縮小不太敏感（第 5 章），所以換成固定除數，在我們的訓練裡不會讓更新的大小差很多；下面的實測會看到。

**難度偏差（除以標準差）**。同一組回答的標準差小，代表這題太容易（幾乎全對）或太難（幾乎全錯），除以它會把這一組的 advantage 放大。8 個回答答對 \\( k \\) 個，整組 advantage 的絕對值加總，不除與除以標準差：

```python
import torch

from llmz.rl import group_advantages

print("k   不除標準差  除以標準差  放大幾倍")
for k in range(1, 8):
    rewards = torch.tensor([[1.0] * k + [0.0] * (8 - k)])
    plain, normed = (group_advantages(rewards, normalize_std=n).abs().sum().item() for n in (False, True))
    print(f"{k}   {plain:8.2f}  {normed:10.2f}  {normed / plain:7.2f}")
```

執行結果：

```text
k   不除標準差  除以標準差  放大幾倍
1       1.75        4.95     2.83
2       3.00        6.48     2.16
3       3.75        7.25     1.93
4       4.00        7.48     1.87
5       3.75        7.25     1.93
6       3.00        6.48     2.16
7       1.75        4.95     2.83
```

答對 1 個或 7 個的那一組放大 2.83 倍，答對 4 個的放大 1.87 倍：相對於不除，除以標準差讓太容易、太難的題分到的權重多了約 1.5 倍。第 3 集看到的「困難題學得快很多」就是這個放大。論文稱它為難度偏差：太容易、太難的題在更新裡權重比較高。另外，一般的 advantage 標準化是對整批做的；一題一題做，才會讓每一題的權重不一樣。

Dr. GRPO 的做法就是兩個都拿掉：`normalize_std=False`、`length_norm=False` 加固定的 `length_divisor`。

### GSPO：比值改成整條回答一個

Dr. GRPO 改的是 advantage 與長度的權重；GSPO（Group Sequence Policy Optimization，以整條回答為單位的 policy 最佳化，Zheng 等人，2025，arXiv 2507.18071）改的是比值 \\( \rho \\)。論文的論點：重要性取樣（第 4 集的 \\( \rho \\)）要靠對同一個分布抽很多個樣本，才能校正取樣的分布與現在的 policy 的差距。GRPO 的 \\( \rho_{i,t} \\) 在每個位置只有一個樣本（取樣到的那一個 token），校正不了什麼，反而把雜訊加進梯度，回答越長累積越多。而獎勵是給整條回答的，所以「超出範圍就不再推」也應該以整條回答為單位。

GSPO 的比值是整條回答的機率比，再取長度的幾何平均（也就是各 token 的 log 比值的平均，再 `exp`）：

\\[ s_i = \left( \frac{\pi_\theta(o_i)}{\pi_{\text{old}}(o_i)} \right)^{1/|o_i|} = \exp\left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \rho_{i,t} \right) \\]

其他部分不變：同一條回答的所有 token 共用 \\( s_i \\)，取 \\( \min(s_i A_i, \text{clip}(s_i, 1\pm\varepsilon) A_i) \\)。不取幾何平均（只加總）的話，長的回答比值會離 1 很遠，不同長度的回答要不同的 clip 範圍；平均之後比值落在同一個尺度（練習）。

我們把它做成 `importance_ratio` 的一個參數：

```python,ignore
{{#include ../../../src/llmz/rl.py:importance_ratio}}
```

`seq_ratio=True` 時，先把被遮掉的位置歸零，對每一列的 log 比值取平均（分母只算 `mask` 內的 token），再 `[:, None].expand_as` 擴張成每個 token 一份。梯度還是會流到每個 token 的 log 機率，每個 token 分到 \\( 1/|o_i| \\)，所有 token 平等：這是論文說的「GSPO 把回答裡的 token 一視同仁」，GRPO 則是每個 token 依自己的 \\( \rho_{i,t} \\) 加權。看三種情況，advantage 都是 +1、`clip_eps=0.2`：

```python
import torch

from llmz.rl import grpo_loss, importance_ratio

cases = [  # (說明, 每個 token 的 log 比值)；advantage 都是 +1，clip_eps = 0.2（比值的範圍 0.8～1.2）
    ("一個 token", [0.3]),
    ("第 1 個 token 比值 e^0.3 = 1.35，第 2 個 1.0", [0.3, 0.0]),
    ("第 1 個 token 比值 e^0.5 = 1.65，第 2 個 e^-0.1 = 0.90", [0.5, -0.1]),
]
for name, log_ratios in cases:
    print(name)
    for seq_ratio in [False, True]:
        logp = torch.tensor([log_ratios], requires_grad=True)
        mask = torch.ones_like(logp, dtype=torch.bool)
        ratio = importance_ratio(logp.detach(), torch.zeros_like(logp), mask, seq_ratio)[0]
        grpo_loss(logp, torch.zeros_like(logp), torch.ones(1), mask, seq_ratio=seq_ratio).backward()
        label = "序列層級" if seq_ratio else "token 層級"
        print(f"  {label}  比值 {[round(r, 3) for r in ratio.tolist()]}  每個 token 的梯度 {[round(g + 0.0, 3) for g in logp.grad[0].tolist()]}")
```

執行結果：

```text
一個 token
  token 層級  比值 [1.35]  每個 token 的梯度 [0.0]
  序列層級  比值 [1.35]  每個 token 的梯度 [0.0]
第 1 個 token 比值 e^0.3 = 1.35，第 2 個 1.0
  token 層級  比值 [1.35, 1.0]  每個 token 的梯度 [0.0, -0.5]
  序列層級  比值 [1.162, 1.162]  每個 token 的梯度 [-0.581, -0.581]
第 1 個 token 比值 e^0.5 = 1.65，第 2 個 e^-0.1 = 0.90
  token 層級  比值 [1.649, 0.905]  每個 token 的梯度 [0.0, -0.452]
  序列層級  比值 [1.221, 1.221]  每個 token 的梯度 [0.0, 0.0]
```

- **只有一個 token**：序列比值就是那個 token 的比值，兩種做法完全一樣（`tests/test_rl.py` 驗證 loss 與梯度都相同，包含 clip 的每一種情況）。
- **第二種**：token 層級把第 1 個 token 擋住（梯度 0），第 2 個照給；序列比值 \\( e^{0.15} = 1.16 \\) 在範圍內，兩個 token 都有梯度，而且一樣大。
- **第三種**：反過來。token 層級只擋住第 1 個 token；序列比值 \\( e^{0.2} = 1.22 \\) 超過 1.2，整條回答一起被擋住。

GSPO 是「以整條回答為單位」清掉過度偏離的樣本，GRPO 是以 token 為單位。論文因此用小得多的 clip 範圍：實驗裡 GSPO 的範圍是 3e-4 與 4e-4，GRPO 是 0.2 與 0.27，被 clip 的 token 比例是 GSPO 約 0.15、GRPO 約 0.0013，差兩個數量級；它們也報告 GSPO 的訓練比 GRPO 有效率、對 MoE 模型特別穩（MoE 一次更新之後，同一條回答啟動的專家會變，token 層級的比值就跳得很兇）。這些是 Qwen3-30B-A3B-Base 的實驗，我們的模型小得多，不會去驗證它們。

### 實測：拿掉兩個除法、換成序列比值

用第 5 集的對照組：從只 SFT 100 步的 `runs/sft_short` 開始，練 `add` 與 `subtract`，學習率 `1e-5`、600 步，種子 0 與 1。只改 loss 的旗標：

```bash
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --out_dir runs/rlvr_short                 # 第 5 集的 GRPO
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --no-normalize_std --out_dir runs/rlvr_short_nostd                      # 不除標準差
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --no-length_norm --length_divisor 4 --out_dir runs/rlvr_short_nolen    # 固定除數（max_new_tokens = 4）
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --no-normalize_std --no-length_norm --length_divisor 4 --out_dir runs/rlvr_short_dr   # Dr. GRPO
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --seq_ratio --out_dir runs/rlvr_short_gspo                          # GSPO
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --seq_ratio --clip_eps 0.02 --out_dir runs/rlvr_short_gspo_eps0.02   # GSPO，clip 範圍縮小到 0.02
```

種子 1 的每一行加 `--seed 1`，`--out_dir` 的名稱後面加 `_seed1`（第 5 集的 `runs/rlvr_short_seed1` 就是這樣來的）。新的十個訓練迴圈各是 309～323 秒。把五種做法、兩個種子在 500 題驗證題上的正確率整理成表（每 50 步驗證一次，這裡取 5 個點；`clip_frac` 與 `grad_norm` 是每 10 步一筆的紀錄的平均）：

```python,norun
import json

variants = [("grpo", "rlvr_short"), ("no_std", "rlvr_short_nostd"), ("no_len", "rlvr_short_nolen"), ("dr_grpo", "rlvr_short_dr"), ("gspo", "rlvr_short_gspo")]
steps = [100, 200, 300, 400, 600]
for seed, suffix in [(0, ""), (1, "_seed1")]:
    print(f"種子 {seed}   " + "".join(f"{s:>7d}" for s in steps) + "   clip_frac  grad_norm")
    for label, name in variants:
        records = [json.loads(line) for line in open(f"runs/{name}{suffix}/log.jsonl")]
        val = {r["step"]: r["val_accuracy"] for r in records if "val_accuracy" in r}
        logged = [r for r in records if "clip_frac" in r]  # 每 10 步記一筆
        clip = sum(r["clip_frac"] for r in logged) / len(logged)
        grad = sum(r["grad_norm"] for r in logged) / len(logged)
        print(f"{label:<9s}  " + "".join(f"{val[s]:7.3f}" for s in steps) + f"   {clip:9.4f}  {grad:9.2f}")
```

輸出（`log.jsonl` 是固定的檔案，這一段每次一樣）：

```text
種子 0       100    200    300    400    600   clip_frac  grad_norm
grpo         0.452  0.652  0.654  0.702  0.702      0.0043       1.06
no_std       0.446  0.548  0.548  0.548  0.548      0.0034       0.30
no_len       0.428  0.582  0.606  0.662  0.662      0.0056       0.51
dr_grpo      0.434  0.494  0.494  0.494  0.548      0.0029       0.15
gspo         0.442  0.494  0.494  0.548  0.548      0.0012       0.86
種子 1       100    200    300    400    600   clip_frac  grad_norm
grpo         0.386  0.534  0.578  0.578  0.578      0.0024       0.67
no_std       0.404  0.506  0.560  0.560  0.626      0.0038       0.29
no_len       0.376  0.468  0.548  0.566  0.578      0.0042       0.33
dr_grpo      0.412  0.506  0.598  0.598  0.650      0.0041       0.18
gspo         0.418  0.548  0.578  0.628  0.628      0.0010       0.83
```

訓練完用第 5 集最後那段評估程式（1000 題 greedy、取樣獎勵、信心），把 `setups` 換成下面這些模型（第一個是只 SFT 100 步的起點）：

```python,ignore
setups = [(("add", "subtract"), [
    "runs/sft_short/model.pt",
    "runs/rlvr_short/model.pt", "runs/rlvr_short_nostd/model.pt", "runs/rlvr_short_nolen/model.pt", "runs/rlvr_short_dr/model.pt", "runs/rlvr_short_gspo/model.pt",
    "runs/rlvr_short_seed1/model.pt", "runs/rlvr_short_nostd_seed1/model.pt", "runs/rlvr_short_nolen_seed1/model.pt", "runs/rlvr_short_dr_seed1/model.pt", "runs/rlvr_short_gspo_seed1/model.pt",
    "runs/rlvr_short_gspo_eps0.02/model.pt", "runs/rlvr_short_gspo_eps0.02_seed1/model.pt",
])]
```

輸出（某一次，取樣的部分每次略有不同）：

```text
runs/sft_short/model.pt      greedy 0.266（add 0.289  subtract 0.246）  取樣獎勵 0.184  信心 0.27
runs/rlvr_short/model.pt     greedy 0.694（add 0.910  subtract 0.513）  取樣獎勵 0.693  信心 0.95
runs/rlvr_short_nostd/model.pt greedy 0.558（add 0.838  subtract 0.324）  取樣獎勵 0.557  信心 0.99
runs/rlvr_short_nolen/model.pt greedy 0.664（add 0.838  subtract 0.518）  取樣獎勵 0.662  信心 0.96
runs/rlvr_short_dr/model.pt  greedy 0.558（add 0.838  subtract 0.324）  取樣獎勵 0.556  信心 0.98
runs/rlvr_short_gspo/model.pt greedy 0.558（add 0.838  subtract 0.324）  取樣獎勵 0.557  信心 0.98
runs/rlvr_short_seed1/model.pt greedy 0.591（add 0.910  subtract 0.324）  取樣獎勵 0.590  信心 0.99
runs/rlvr_short_nostd_seed1/model.pt greedy 0.638（add 0.947  subtract 0.379）  取樣獎勵 0.637  信心 0.97
runs/rlvr_short_nolen_seed1/model.pt greedy 0.591（add 0.910  subtract 0.324）  取樣獎勵 0.589  信心 0.99
runs/rlvr_short_dr_seed1/model.pt greedy 0.655（add 0.875  subtract 0.471）  取樣獎勵 0.652  信心 0.98
runs/rlvr_short_gspo_seed1/model.pt greedy 0.631（add 0.932  subtract 0.379）  取樣獎勵 0.629  信心 0.99
runs/rlvr_short_gspo_eps0.02/model.pt greedy 0.558（add 0.838  subtract 0.324）  取樣獎勵 0.558  信心 0.99
runs/rlvr_short_gspo_eps0.02_seed1/model.pt greedy 0.634（add 0.737  subtract 0.548）  取樣獎勵 0.630  信心 0.96
```

整理成表（1000 題 greedy，單次；每一格「左／右」是種子 0／種子 1）：

| | 合計 | `add` | `subtract` |
| --- | --- | --- | --- |
| 起點（`runs/sft_short`） | 0.266 | 0.289 | 0.246 |
| GRPO（第 5 集） | 0.694／0.591 | 0.910／0.910 | 0.513／0.324 |
| 不除標準差 | 0.558／0.638 | 0.838／0.947 | 0.324／0.379 |
| 固定除數 | 0.664／0.591 | 0.838／0.910 | 0.518／0.324 |
| Dr. GRPO | 0.558／0.655 | 0.838／0.875 | 0.324／0.471 |
| GSPO | 0.558／0.631 | 0.838／0.932 | 0.324／0.379 |
| GSPO，`clip_eps` 0.02 | 0.558／0.634 | 0.838／0.737 | 0.324／0.548 |

改過 `grpo_loss` 之後，我們用第 5 集的指令（加上 `--max_steps 20`）重新跑了 GRPO 的前 20 步，每一筆紀錄的 `reward`、`loss`、`clip_frac`、`mixed_groups`、`resp_len`、`grad_norm` 都和第 5 集的 `runs/rlvr_short/log.jsonl` 一樣：預設值的行為沒變。上面 GRPO 那兩列（0.694 與 0.591）就是第 5 集的那兩個模型，用同一段評估程式再量一次。我們從這些數字看到的：

- **看不出哪一種比較好**。同一個種子內，種子 0 是 GRPO 最高（0.694，其他 0.558～0.664），種子 1 卻是 GRPO 最低（0.591，和固定除數並列；其他 0.591～0.655）：誰好誰壞隨種子對調。同一個做法換個種子就差 0.07～0.10（GRPO 0.694 對 0.591，Dr. GRPO 0.558 對 0.655），不比做法之間的差距小。500 題的驗證曲線也是這樣：最後的正確率在 0.548～0.702 之間，沒有一種做法在兩個種子上都贏。要比出哪一種比較好，要更多的種子，或更大的設定。同一個種子的不同做法，也常常停在一模一樣的 1000 題結果（種子 0 的不除標準差、Dr. GRPO、GSPO 三個都是 0.558，`add` 0.838、`subtract` 0.324）：我們猜模型走到哪裡，主要由種子決定的題目與取樣順序決定，旗標只是小幅度的推動；這一點沒有進一步驗證。
- **長度正規化在這裡沒有東西可以影響**。所有的訓練裡 `resp_len` 都是 2.00：回答是一個字加結尾，每條都是 2 個 token。token 加總再除以 4，就是 token 平均再乘 0.5，各 token 之間的相對權重完全沒變。`grad_norm` 的平均從 1.06 變成 0.51（種子 0），剛好一半。種子 1 用固定除數訓練出來的模型，1000 題的結果和 GRPO 的一模一樣（0.591、`add` 0.910、`subtract` 0.324）。要看到 Dr. GRPO 說的長度偏差，回答要有長有短，我們的題目做不到，這個偏差只能用上面第一個程式看。
- **不除標準差，梯度小得多，曲線卻沒變慢**。`grad_norm` 從 0.7～1.06 掉到 0.3，100 步的正確率卻是 0.446 對 0.452、0.404 對 0.386。第 3 集的玩具題用普通的梯度上升，步伐變小就是學得慢；這裡的 `AdamW` 把整體大小的差異吸收了，剩下的只是各組之間的相對權重（難度偏差），在我們的設定裡看不出它的影響。
- **GSPO 的 `clip_frac` 比較低，把 `clip_eps` 縮小它就上來了**。同樣的 `clip_eps=0.2`，每個 token 的比值被擋住的比例：GRPO 是 0.0024～0.0043，GSPO 是 0.0010～0.0012。回答只有 2 個 token，序列比值是兩個 token 比值的幾何平均，比單一 token 少超出範圍。論文用小得多的範圍，所以我們也試了 `--clip_eps 0.02`（上面最後一個指令）：`clip_frac` 的平均變成 0.038（種子 0）與 0.135（種子 1），是 `0.2` 的幾十到一百多倍，1000 題的正確率是 0.558 與 0.634，和 `0.2` 的 0.558 與 0.631 沒有差別。論文說擋掉那麼多 token 沒有讓訓練變差；我們也沒有看到變差，但也沒有看到變好。
- **這些實驗證實不了、也推翻不了論文**。論文的實驗是 30B 的 MoE 模型、很長的回答；我們是 2 個 token 的回答、一個小模型。我們只能說：在這個設定裡，這幾個旗標對正確率沒有看得出來的影響。

## 重點整理

- GRPO 有兩個除法：除以組內標準差（`group_advantages` 的 `normalize_std`）、除以回答的 token 數（`grpo_loss` 的 `length_norm`）。Dr. GRPO（Liu 等人，2025）主張拿掉，因為它們各帶來一種偏差：長度偏差（答錯的長回答，每個 token 被壓得比較輕，模型寧可寫長）與難度偏差（太容易、太難的題，advantage 被放大）。對應 `normalize_std=False`，加 `length_norm=False` 搭配固定的 `length_divisor`（例如 `max_new_tokens`）。
- GSPO（Zheng 等人，2025）把每個 token 一個比值，換成整條回答一個：各 token 的 log 比值取平均再 `exp`，也就是幾何平均（`importance_ratio(..., seq_ratio=True)`），clip 也以整條回答為單位。回答只有一個 token 時，兩種做法完全一樣（`tests/test_rl.py`）。論文用的 clip 範圍（3e-4 與 4e-4）比 GRPO 的（0.2 與 0.27）小得多。
- 序列比值要除以長度（取平均，不取加總）：不除的話，比值隨長度指數成長，長的回答幾乎一定被擋住（練習）。
- M4 Max 上，`runs/sft_short` 練 `add` 與 `subtract`，五種做法、兩個種子：1000 題 greedy 是 0.558～0.694，看不出哪一種比較好：同一個做法的兩個種子差 0.07～0.10，不比做法之間的差距小。我們的回答都是 2 個 token，`length_norm` 只差一個常數倍（`grad_norm` 剛好減半）；不除標準差讓梯度小了 2～3.5 倍，`AdamW` 把它吸收了；GSPO 在 `clip_eps=0.2` 的 `clip_frac` 是 0.001，縮小到 0.02 之後是 0.04～0.14，正確率沒有變。
