# RLVR：用可驗證獎勵訓練 SFT 模型

## 本集目標

把上一集的 loss 接到第 14 章的 SFT 模型：每一步對一批題目各取樣一組回答、用 `reward` 打 0／1、算 group advantage、更新。量出 `color` 與 `who` 兩種題型的正確率，在訓練前後差多少；再換一個「有時候會做對」的起點，看 RLVR 什麼時候有用、什麼時候沒用。

## 正文

### 要訓練什麼

第 14 章的 SFT 模型（`runs/sft/model.pt`）在加、減、數數量上全對，`color`（兩個顏色選一個）約 1/2、`who`（三個人選一個）約 1/3，剛好是亂猜的機率。全對的題型沒有進步的空間（一組回答全對，advantage 是 0），所以這一集只訓練 `color` 與 `who`。

這種做法叫 RLVR（reinforcement learning with verifiable rewards，用可驗證的獎勵做強化學習）：獎勵不是另一個模型打的分數，是程式直接檢查的結果。第 14 章第 1 集的 `reward(task, reply)` 就是：回覆和標準答案一樣是 1、不一樣是 0。

### rollout：取樣一組回答，留下 token id

GRPO 需要每個回答的 token id（算 `old_logp`、算比值）和哪些 token 屬於回答（`mask`）。第 14 章的 `generate_replies` 只回傳字串，所以我們另外寫 `rollout`：

```python,ignore
{{#include ../../../src/llmz/rl.py:rollout}}
```

- **回傳**：`ids` 是提示加回答的 token id，`mask` 只有回答的 token 是 `True`（到第一個 `<|endoftext|>` 為止，含它：模型要學會什麼時候停）；`rewards` 的形狀是 `(題數, group_size)`，正好是 `group_advantages` 要的 `(P, G)`。
- **同一題的 G 個回答怎麼來**：把同一個提示重複 G 次放進同一個 batch，用 `temperature=1.0` 抽樣。`1.0` 是照模型本來的機率抽，這時 `old_logp` 才真的是「取樣時的機率」。
- **為什麼不用 `generate_cached`**：第 11 章的 `pick_next` 為了支援 `top_p`，每一列各抽一次，在 MPS 上每一列都是好幾個小運算。rollout 一次要抽 128 條回答，時間幾乎都花在這裡。`sample_batch` 用同一套 KV cache，只把抽樣換成整個 batch 一次抽（沒有 `top_p`）。量一下：

```python,norun
import torch

from llmz.device import get_device
from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import encode_chat, make_tasks, task_messages
from llmz.profile import timeit
from llmz.rl import rollout, sample_batch
from llmz.tokenizer import Tokenizer

device = get_device()
tokenizer = Tokenizer.load("data/tokenizer.json")
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/sft/model.pt", map_location=device))

tasks = make_tasks(16, seed=0, kinds=("color", "who"))
prompts = [encode_chat(tokenizer, task_messages(t), generation_prompt=True)[0] for t in tasks for _ in range(8)]  # 16 題 * 8 個回答 = 128 條
by_length = {}
for p in prompts:
    by_length.setdefault(len(p), []).append(p)  # 長度相同的放同一個 batch
print(f"{len(prompts)} 條提示，{len(by_length)} 種長度，每條抽 4 個 token")


def generate_all(generate):
    for batch in by_length.values():
        generate(model, torch.tensor(batch, device=device), 4, 1.0)


for name, generate in [("generate_cached", generate_cached), ("sample_batch", sample_batch)]:
    print(f"{name:16s} {timeit(lambda: generate_all(generate), device, warmup=1, repeat=3):.2f} 秒")
print(f"整個 rollout     {timeit(lambda: rollout(model, tokenizer, tasks, 8, 4), device, warmup=1, repeat=3):.2f} 秒（含編碼、打分數）")
```

M4 Max 上的輸出（某一次）：

```text
128 條提示，2 種長度，每條抽 4 個 token
generate_cached  3.79 秒
sample_batch     0.15 秒
整個 rollout     0.16 秒（含編碼、打分數）
```

同一段程式跑了兩次：`generate_cached` 是 3.8～5.0 秒，`sample_batch` 是 0.15～0.25 秒，大約差 20 倍。換成 `generate_cached`，一步 RLVR 會從約 0.6 秒（下面的完整訓練）變成 4～5 秒：沒有針對取樣最佳化的 rollout，是 RL 的瓶頸；第 7 集會看真實系統怎麼處理它。`tests/test_rl.py` 驗證：溫度很低時 `sample_batch` 的結果和 `generate_cached` 的 greedy 一樣，而且 4000 列同一個提示抽出的第一個 token，頻率符合 softmax。

### 一步 RLVR

```python,ignore
{{#include ../../../src/llmz/rl.py:train_rlvr}}
```

每一步（迴圈裡的 `for step`）：

1. **rollout**：從訓練題裡取 `prompts_per_step`（16）題，每題抽 `group_size`（8）個回答，一共 128 條。
2. **獎勵與 advantage**：`rollout` 已經用 `reward` 打好 0／1，`group_advantages` 照第 3 集的方法算組內的 advantage（預設除以組內標準差），攤平成每條回答一個數字。
3. **`old_logp`**：用現在的模型、不帶梯度，算一次每個 token 的 log 機率。
4. **更新 `epochs` 次**：每次重算 `logp`、`grpo_loss`、`backward()`、裁剪梯度、`AdamW` 的 `step()`。第 1 次 `logp` 和 `old_logp` 一樣，比值恰好是 1；第 2 次模型已經動過，比值才不是 1，clip 才有機會用到。
5. **記錄**：`reward` 是這一批 rollout 的平均獎勵；`clip_frac` 是最後一次更新前、比值超出 \\( [1-\varepsilon, 1+\varepsilon] \\) 的 token 比例；`mixed_groups` 是「組內有對有錯」的組的比例，只有這些組的 advantage 不是 0，才有梯度；`resp_len` 是每條回答平均幾個 token（含結尾）。

題目每一步都是新抽的（`make_tasks` 一次產生整個訓練過程要用的題目，一步用 16 題）；驗證題固定（種子 1 的 `color` 與 `who` 的 `"valid"` 題，500 題），所以兩個種子、不同學習率的 `val_accuracy` 才能比。和第 14 章一樣，`val_accuracy` 是 greedy 回答的答對比例。

這裡沒有 KL 懲罰、沒有 reference 模型：只有一個模型在更新。原論文的 GRPO 多了一項把 policy 拉回 reference 模型附近的 KL（第 14 章第 5 集的 RLHF 目標就有這一項）；我們的訓練只有幾百步，先不加，後面會看到少了它的影響。

### smoke 設定

和前面的章節一樣，先用 CPU 上幾秒的縮小版確認程式跑得通：byte-level tokenizer、隨機初始化的小模型、5 步：

```python
from llmz.model import ModelConfig
from llmz.rl import RLVRConfig, train_rlvr
from llmz.tokenizer import Tokenizer, base_vocab

END = "<|endoftext|>"
tokenizer = Tokenizer(base_vocab([END]), [], [END])  # 沒有 merges：一個 byte 一個 token
cfg = RLVRConfig(
    sft_path="", out_dir="smoke_rlvr",  # 空字串：不載入 SFT 模型，隨機初始化
    model=ModelConfig(vocab_size=257, context_length=256, d_model=64, n_layers=2, n_heads=4, d_ff=128),
    prompts_per_step=4, group_size=4, max_new_tokens=4, max_steps=5, log_every=1, eval_every=5, n_valid=20, device="cpu",
)
records = train_rlvr(cfg, tokenizer)
print("所有回答都答錯、獎勵全是 0：", all(r["reward"] == 0 for r in records[1:]))
print("沒有訊號，梯度是 0：", all(r["grad_norm"] == 0 for r in records[1:]))
```

執行結果（某一次）：

```text
{'step': 0, 'val_accuracy': 0.0}
{'step': 1, 'reward': 0.0, 'loss': 0.0, 'clip_frac': 0.0, 'mixed_groups': 0.0, 'resp_len': 4.0, 'grad_norm': 0.0, 'time': 0.2}
{'step': 2, 'reward': 0.0, 'loss': 0.0, 'clip_frac': 0.0, 'mixed_groups': 0.0, 'resp_len': 3.94, 'grad_norm': 0.0, 'time': 0.2}
{'step': 3, 'reward': 0.0, 'loss': 0.0, 'clip_frac': 0.0, 'mixed_groups': 0.0, 'resp_len': 4.0, 'grad_norm': 0.0, 'time': 0.3}
{'step': 4, 'reward': 0.0, 'loss': 0.0, 'clip_frac': 0.0, 'mixed_groups': 0.0, 'resp_len': 4.0, 'grad_norm': 0.0, 'time': 0.3}
{'step': 5, 'reward': 0.0, 'loss': 0.0, 'clip_frac': 0.0, 'mixed_groups': 0.0, 'resp_len': 4.0, 'grad_norm': 0.0, 'time': 0.4, 'val_accuracy': 0.0}
所有回答都答錯、獎勵全是 0： True
沒有訊號，梯度是 0： True
```

隨機初始化的模型一題也答不對，所有回答的獎勵都是 0，整批 advantage 是 0、`grad_norm` 是 0，模型一步也沒動。這就是 RL 之前要先 SFT 的原因：**RLVR 只能放大「模型有時候會做對」的行為；從來不會做對的事，沒有訊號可以放大。**`tests/test_rl.py` 另外用一個一開始約一半會拿到的獎勵，確認獎勵真的會往上走。

### 完整訓練：`color` 與 `who`

```bash
uv run tools/rlvr.py                                       # 預設：runs/sft/model.pt 開始、color 與 who、600 步、種子 0
uv run tools/rlvr.py --out_dir runs/rlvr_seed1 --seed 1    # 另一個種子
```

預設設定：每步 16 題、每題 8 個回答、每批 rollout 更新 2 次、學習率 `3e-5`（固定）、`clip_eps=0.2`、`temperature=1.0`、600 步，沒有 KL。M4 Max 上，兩個種子的訓練迴圈各是 381 秒與 386 秒（約 6.4 分鐘，含 13 次驗證；每步約 0.6 秒），遠低於 30 分鐘。結果存在 `runs/rlvr/`（`model.pt`、`log.jsonl`、`config.json`）。同一個種子的兩次獨立執行，前 200 步的 `val_accuracy` 一個一個相同（MPS 上也是），所以下面的數字用同樣的指令可以重現。

`log.jsonl` 裡每 50 步有一次驗證（500 題 greedy，單次）：

| 步數 | 0 | 100 | 200 | 300 | 400 | 500 | 600 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 種子 0 | 0.400 | 0.356 | 0.410 | 0.420 | 0.414 | 0.432 | 0.430 |
| 種子 1 | 0.400 | 0.406 | 0.410 | 0.412 | 0.438 | 0.440 | 0.410 |

在 0.36～0.44 之間來回，500 題的雜訊是 ±2 個百分點左右。訓練結束後，用 1000 題驗證題（上面那 500 題加另外 500 題）仔細量一次，順便看兩個和正確率無關的數字：「取樣獎勵」是 `temperature=1.0` 每題取樣 4 次的平均獎勵，「信心」是第一個回答 token 的最大機率（模型對自己的答案多有把握）的平均：

```python,norun
import torch

from llmz.device import get_device
from llmz.functional import softmax
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import accuracy, encode_chat, make_tasks, task_messages
from llmz.rl import rollout
from llmz.tokenizer import Tokenizer

device = get_device()
tokenizer = Tokenizer.load("data/tokenizer.json")


def confidence(model, tasks):
    """第一個回答 token 的最大機率，對所有題平均：越接近 1，模型對每一題越有把握。"""
    total = 0.0
    for task in tasks:
        ids = torch.tensor([encode_chat(tokenizer, task_messages(task), generation_prompt=True)[0]], device=device)
        with torch.no_grad():
            total += softmax(model(ids)[0, -1]).max().item()
    return total / len(tasks)


setups = [
    (("color", "who"), ["runs/sft/model.pt", "runs/rlvr/model.pt", "runs/rlvr_seed1/model.pt"]),
    (("add", "subtract"), ["runs/sft_short/model.pt", "runs/rlvr_short/model.pt", "runs/rlvr_short_seed1/model.pt"]),
]
for kinds, paths in setups:
    valid = make_tasks(1000, seed=1, split="valid", kinds=kinds)
    for path in paths:
        model = TransformerLM(ModelConfig()).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        torch.manual_seed(0)
        sampled = rollout(model, tokenizer, valid, 4, 4).rewards.mean().item()  # temperature 1.0 取樣 4 次的平均獎勵
        per_kind = "  ".join(f"{kind} {accuracy(model, tokenizer, [t for t in valid if t.kind == kind]):.3f}" for kind in kinds)
        print(f"{path:28s} greedy {accuracy(model, tokenizer, valid):.3f}（{per_kind}）  取樣獎勵 {sampled:.3f}  信心 {confidence(model, valid):.2f}")
```

M4 Max 上的輸出（某一次，取樣的部分每次略有不同）：

```text
runs/sft/model.pt            greedy 0.401（color 0.491  who 0.304）  取樣獎勵 0.400  信心 0.56
runs/rlvr/model.pt           greedy 0.426（color 0.534  who 0.310）  取樣獎勵 0.419  信心 0.95
runs/rlvr_seed1/model.pt     greedy 0.418（color 0.495  who 0.335）  取樣獎勵 0.421  信心 0.97
runs/sft_short/model.pt      greedy 0.266（add 0.289  subtract 0.246）  取樣獎勵 0.184  信心 0.27
runs/rlvr_short/model.pt     greedy 0.694（add 0.910  subtract 0.513）  取樣獎勵 0.693  信心 0.95
runs/rlvr_short_seed1/model.pt greedy 0.591（add 0.910  subtract 0.324）  取樣獎勵 0.590  信心 0.99
```

`color` 與 `who` 的前後（1000 題 greedy，單次）：

| | 合計 | `color` | `who` | 取樣獎勵 | 信心 |
| --- | --- | --- | --- | --- | --- |
| SFT（`runs/sft`） | 0.401 | 0.491 | 0.304 | 0.400 | 0.56 |
| RLVR 種子 0 | 0.426 | 0.534 | 0.310 | 0.419 | 0.95 |
| RLVR 種子 1 | 0.418 | 0.495 | 0.335 | 0.421 | 0.97 |

合計正確率從 0.401 變成 0.418～0.426：兩個種子都高一點，但 1000 題的雜訊約 ±1.5 個百分點（兩個種子各高 1.7 與 2.5 個百分點，約一到兩個標準差），而且 `color` 一個進步 4 個百分點、一個沒動，`who` 兩個都在 1/3 附近。我們不敢說這是進步。換成別的 1000 道題，`color` 會再差幾個百分點（下面「副作用」那張表裡，同一個模型的 `color` 是 0.503）。取樣獎勵也只從 0.400 到 0.42。**RLVR 訓練 600 步，沒有讓模型學會 `color` 與 `who`。**

變的是「信心」：0.56 變成 0.95～0.97。模型對每一題的第一個回答 token，現在幾乎都有 95% 以上的把握，只是這個把握和答案對不對無關。

### 為什麼沒進步：group 裡不再有對有錯

回頭看訓練紀錄裡的 `mixed_groups`（組內有對有錯的比例），每一段取平均：

| 步數 | 1～100 | 101～200 | 201～400 | 401～600 |
| --- | --- | --- | --- | --- |
| 種子 0 | 0.46 | 0.21 | 0.16 | 0.13 |
| 種子 1 | 0.33 | 0.23 | 0.07 | 0.07 |

一開始 SFT 模型對同一題的回答不固定，一組 8 個常常有對有錯（本集的練習量過一次：`color` 有 97%、`who` 有 91% 的組是這樣）；訓練一陣子之後，同一題的 8 個回答越來越常完全一樣，這時整組 advantage 是 0，沒有梯度（第 3 集）。同一個提示的答案變得很確定，但對不對是各題各自的運氣。我們的解讀（沒有直接驗證）：模型沒有「該答哪一個」的線索可以放大，獎勵只是把它推向其中一個答案；亂猜的期望值不會因此改變。

學習率會影響這件事發生得多快。其他學習率，同一個種子 0、200 步（`--lr` 與 `--max_steps` 不同，其他一樣；`mixed_groups` 取第 101～200 步的平均）：

| `--lr` | 0 步 | 50 步 | 100 步 | 150 步 | 200 步 | `mixed_groups` |
| --- | --- | --- | --- | --- | --- | --- |
| `1e-5` | 0.400 | 0.424 | 0.420 | 0.406 | 0.422 | 0.62 |
| `3e-5`（預設） | 0.400 | 0.368 | 0.356 | 0.374 | 0.410 | 0.21 |
| `1e-4` | 0.400 | 0.408 | 0.392 | 0.354 | 0.364 | 0.10 |
| `3e-4` | 0.400 | 0.272 | 0.278 | 0.198 | 0.200 | 0.02 |

```bash
uv run tools/rlvr.py --lr 1e-4 --max_steps 200 --out_dir runs/rlvr_lr1e-4   # 其餘類推
```

小的學習率（`1e-5`）組內還有對有錯，但 200 步內正確率沒有動；大的學習率很快就把組內變成一致，`3e-4` 更把正確率推到 0.20，比亂猜（0.40）還低。每一個學習率都沒有讓正確率往上。

SFT 的結果是類似的：第 14 章把 SFT 的訓練量加到 20 倍，`color` 與 `who` 也停在亂猜。兩者的共同點是：這個模型在這個預算裡沒有學會「找出被問的那樣東西」。RLVR 的獎勵再乾淨，也只能放大模型已經有時候會做對的事；沒有可以放大的東西，它就只會讓模型更確定自己原本的猜法。（為什麼學不會，我們沒有進一步查。）

### 副作用：沒練的題型也變了

這個 RLVR 只用 `color` 與 `who` 的題目、沒有 KL 把模型拉在原來的附近。看 1000 題五種題型都有的驗證題，第 14 章的做法（greedy）：

| | `add` | `subtract` | `count` | `color` | `who` | 合計 |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 188／188 | 200／200 | 192／192 | 101／199 | 74／221 | 755／1000 |
| RLVR 種子 0 | 188／188 | 160／200 | 147／192 | 100／199 | 60／221 | 655／1000 |
| RLVR 種子 1 | 188／188 | 198／200 | 141／192 | 100／199 | 70／221 | 697／1000 |

`color` 與 `who` 沒進步，原本全對的 `subtract` 與 `count` 卻掉了：`count` 兩個種子都從 192／192 掉到 141～147，`subtract` 一個掉到 160／200、一個只掉 2 題。我們沒有查是什麼機制，只知道一個只看 `color` 與 `who` 獎勵的 600 步更新，會動到模型裡負責別的題型的部分。第 14 章第 5 集的 RLHF 目標有 KL 項，原論文的 GRPO 也一樣，把 policy 拉回 reference 模型（這裡就是 `runs/sft`）附近，目的之一就是防這個；我們沒有實作，也沒有試加了之後會不會比較好，這是目前的限制。

### 對照：模型有時候會做對的時候

上面是「沒有東西可以放大」的情況。換一個「有時候會做對」的起點看看。SFT 只訓練 100 步（原本是 1000 步）：

```bash
uv run tools/sft.py --out_dir runs/sft_short --max_steps 100
```

6.5 秒的訓練迴圈。這個模型已經學到格式，`count` 是全對（192／192），但加法與減法還沒學好：1000 題裡 `add` 62／188、`subtract` 53／200。SFT 學得多一點（`--max_steps 200`）時這兩種會升到 150／188 與 83／200，到 1000 步全對：能力是練得出來的，只是這個模型還沒練到。用 RLVR 練 `add` 與 `subtract`：

```bash
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --out_dir runs/rlvr_short
uv run tools/rlvr.py --sft_path runs/sft_short/model.pt --kinds add,subtract --lr 1e-5 --out_dir runs/rlvr_short_seed1 --seed 1
```

訓練迴圈各 313 秒。學習率換成 `1e-5`：`3e-5` 與 `1e-4`（150 步）時組內很快就沒有對有錯，正確率停在 0.536 與 0.430，`1e-5` 可以一路爬。500 題驗證（greedy，單次）：

| 步數 | 0 | 100 | 200 | 300 | 400 | 600 |
| --- | --- | --- | --- | --- | --- | --- |
| 種子 0 | 0.274 | 0.452 | 0.652 | 0.654 | 0.702 | 0.702 |
| 種子 1 | 0.274 | 0.386 | 0.534 | 0.578 | 0.578 | 0.578 |

1000 題驗證（上面 `evalall` 輸出的後三行）：

| | 合計 | `add` | `subtract` | 取樣獎勵 | 信心 |
| --- | --- | --- | --- | --- | --- |
| 短 SFT（`runs/sft_short`） | 0.266 | 0.289 | 0.246 | 0.184 | 0.27 |
| RLVR 種子 0 | 0.694 | 0.910 | 0.513 | 0.693 | 0.95 |
| RLVR 種子 1 | 0.591 | 0.910 | 0.324 | 0.590 | 0.99 |

0.266 變成 0.591～0.694。`add` 兩個種子都是 0.910（原來 0.289），`subtract` 差很多（0.513 與 0.324，原來 0.246）：同一個設定，兩個種子的結果可以差 10 個百分點，RL 的結果雜訊很大，要看兩次以上才能下結論。

這兩條曲線也停了：種子 0 從 350 步、種子 1 從 300 步起，驗證正確率到 600 步都沒有變。原因和上面一樣：最後 200 步的 `mixed_groups` 只剩 0.03 與 0.02，幾乎每一組都是整組全對或整組全錯。簡單的題已經全對，難的題（`subtract` 的某些情況）整組全錯，沒有訊號，第 3 集算過：答對機率 1% 的題，8 個回答有 92% 的機會全錯。

副作用也在：沒練的 `count` 從 192／192 掉到 110～122；`color` 與 `who` 仍在亂猜附近（115／199 與 62～65／221）。

### RLVR 放大的是什麼

兩個實驗放在一起：

- 模型有時候會做對（`add`、`subtract` 在 SFT 100 步時有兩三成），RLVR 讓它變成大部分時候做對；
- 模型從來沒有可以依靠的線索（`color`、`who`），RLVR 只讓它更確定、更一致，正確率不變；
- 全部做對或全部做錯的題，不提供訊號；
- 沒練的題型也會跟著變：這次沒有 KL、沒有 reference 模型，它們退步了（我們沒有試加 KL）。

這和第 14 章的結論一致：後訓練是把模型已經有的能力引出來、放大，不是憑空長出新能力。

## 重點整理

- RLVR 的一步：`rollout` 對每題抽一組回答（保留 token id 與 `mask`）→ `reward` 打 0／1 → `group_advantages` → `old_logp` → 對同一批更新 `epochs` 次的 `grpo_loss`。程式在 `llmz.rl`（`rollout`、`sample_batch`、`RLVRConfig`、`train_rlvr`），命令列是 `tools/rlvr.py`。
- `sample_batch` 用第 11 章的 KV cache、整個 batch 一次抽樣；`generate_cached` 每列各抽一次，在 M4 Max 上 128 條回答要 3.8～5.0 秒，`sample_batch` 是 0.15～0.25 秒。
- 隨機初始化的模型一題也答不對，獎勵全是 0、梯度是 0：RLVR 之前要先 SFT。
- SFT 模型上訓練 `color` 與 `who`（M4 Max 約 6.4 分鐘、600 步、兩個種子）：合計正確率 0.401 變成 0.418～0.426，在雜訊的範圍內，沒有學會；`mixed_groups` 從 0.3～0.5 掉到 0.07～0.13，模型的把握從 0.56 升到 0.95 以上；原本全對的 `count` 掉到 141～147／192。
- SFT 只練 100 步、再用 RLVR 練 `add` 與 `subtract`（學習率 `1e-5`、每個種子約 313 秒）：正確率 0.266 變成 0.591～0.694，兩個種子差 10 個百分點；兩條曲線在 300～350 步停住，因為組內不再有對有錯。
- RLVR 放大模型已經有時候會做對的事；沒有可以放大的東西時，只會讓模型更確定；整組全對或全錯的題沒有訊號；沒練的題型也會跟著變（這次的設定下退步了）。
