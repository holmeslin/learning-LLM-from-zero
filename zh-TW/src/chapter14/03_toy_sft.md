# SFT 玩具版

## 本集目標

在第 5 章的標準模型上做一次 SFT：拿程式產生的（指令，回覆）排成 chat template，只在回覆上算 loss，訓練約一分鐘。量出前後的正確率，看它學到了什麼、沒學到什麼。

## 正文

### 訓練迴圈

SFT 的訓練迴圈和第 5 章第 8 集的 `train` 是同一個：算學習率、取 batch、forward、loss、`backward()`、裁剪、`step()`、`zero_grad()`。三個地方不一樣：

```python,ignore
{{#include ../../../src/llmz/posttrain.py:train_sft}}
```

- **起點**：從 `base_path`（預設 `runs/base/model.pt`，第 5 章的標準模型）載入參數，不是隨機初始化。
- **資料**：不是從 token 檔隨機切一段，而是 `make_tasks` 產生的題目，經過 `encode_chat` 變成 `(ids, mask)`，每步隨機抽 `batch_size` 條、用 `pad_batch` 疊成矩陣。loss 換成上一集的 `sft_loss`。
- **學習率**：最大值 `1e-4`，是預訓練（`1e-3`）的十分之一。微調是在已經練好的參數上輕輕推，推得太大會把原本會的東西沖掉。

驗證用的是 `"valid"` 題（第 1 集：和訓練題絕對不重複）。`val_loss` 一律只算回覆的 token，`val_accuracy` 是 greedy 回答這些題、用 `reward` 打分的平均。`mask=False` 是後面「不套 mask 會怎樣」的對照實驗用的，先不管它。

### smoke 設定

全書規則：訓練的東西都附一個 CPU 上幾秒鐘能跑完的縮小版。這裡用第 3 章的 byte-level tokenizer 加一個隨機初始化的小模型，60 步：

```python
from llmz.model import ModelConfig
from llmz.posttrain import SFTConfig, train_sft
from llmz.tokenizer import Tokenizer, base_vocab

END = "<|endoftext|>"
tokenizer = Tokenizer(base_vocab([END]), [], [END])  # 沒有 merges：一個 byte 一個 token
cfg = SFTConfig(
    base_path="", out_dir="smoke_sft",  # 空字串：不載入，從隨機初始化開始
    model=ModelConfig(vocab_size=257, context_length=256, d_model=64, n_layers=2, n_heads=4, d_ff=128),
    n_train=1000, n_valid=20, batch_size=16, max_steps=60, lr_max=3e-3, lr_min=3e-4, warmup_steps=5,
    log_every=20, eval_every=60, device="cpu",
)
records = train_sft(cfg, tokenizer)
print("val_loss 降到 2 以下：", records[-1]["val_loss"] < 2.0)
```

執行結果（某一次）：

```text
{'step': 20, 'loss': 1.6889, 'lr': 0.002534061990826135, 'grad_norm': 0.939, 'time': 0.6}
{'step': 40, 'loss': 1.3878, 'lr': 0.0010891897324474536, 'grad_norm': 1.06, 'time': 1.3}
{'step': 60, 'loss': 0.9309, 'lr': 0.0003, 'grad_norm': 0.673, 'time': 1.6, 'val_loss': 1.2425, 'val_accuracy': 0.2}
val_loss 降到 2 以下： True
```

沒訓練的模型 loss 約 \\( \ln 257 = 5.5 \\)。回覆只有二十幾種答案，60 步就學會「只回答一個字」，loss 降到 2 以下；但這個小模型從隨機開始，答案對不對還談不上（準確率落在亂猜附近）。`tests/test_posttrain.py` 跑的就是這個設定。

### 完整訓練

完整的版本用命令列：

```bash
uv run tools/sft.py
```

預設是 `n_train=20000`、`batch_size=32`、1000 步、`lr_max=1e-4`。M4 Max 上跑三次，訓練迴圈各是 63～65 秒（含 3 次驗證；較小的晶片會更慢）。存在 `runs/sft/`：`model.pt`（參數，用法和 `runs/base/model.pt` 完全一樣）、`log.jsonl`、`config.json`。其中一次的紀錄（單次）：

| 步數 | 訓練 loss | `val_loss` | `val_accuracy`（200 題） |
| --- | --- | --- | --- |
| 250 | 0.318 | 0.310 | 0.715 |
| 500 | 0.252 | 0.228 | 0.780 |
| 750 | 0.140 | 0.222 | 0.740 |
| 1000 | 0.072 | 0.220 | 0.725 |

`val_loss` 在 500 步之後就幾乎不動；200 題的準確率在 71.5%～78% 之間來回，這是題數太少的雜訊（每題 0.5%）。要看前後的差別，我們用 1000 題。

### 前後的正確率

```python,norun
import torch

from llmz.device import get_device
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import KINDS, generate_replies, make_tasks, reward, task_messages
from llmz.tokenizer import Tokenizer

device = get_device()
tokenizer = Tokenizer.load("data/tokenizer.json")
valid = make_tasks(1000, seed=1, split="valid")
prompts = [task_messages(t) for t in valid]  # 只給使用者那一則，後面接 Assistant: 請模型寫

for path in ["runs/base/model.pt", "runs/sft/model.pt"]:
    model = TransformerLM(ModelConfig()).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    replies = generate_replies(model, tokenizer, prompts)  # greedy，最多 8 個 token，寫出 <|endoftext|> 就停
    hits = {kind: [reward(t, r) for t, r in zip(valid, replies) if t.kind == kind] for kind in KINDS}
    row = "  ".join(f"{kind} {int(sum(v))}/{len(v)}" for kind, v in hits.items())
    total = int(sum(sum(v) for v in hits.values()))
    print(f"{path}：{row}  合計 {total}/{len(valid)} = {total / len(valid):.1%}")
```

輸出（greedy 沒有隨機性；CPU 與 MPS 相同，但每次訓練出來的 `model.pt` 略有不同）：

```text
runs/base/model.pt：add 0/188  subtract 0/200  count 0/192  color 0/199  who 0/221  合計 0/1000 = 0.0%
runs/sft/model.pt：add 188/188  subtract 200/200  count 192/192  color 101/199  who 74/221  合計 755/1000 = 75.5%
```

SFT 前 0/1000（和第 1 集一樣：base model 套上 chat template 也不會回答），SFT 之後 755/1000。三種題型全對，兩種題型停在亂猜。

### 學到了什麼、沒學到什麼

- **格式學到了**。1000 題的回覆全部是一個字，而且寫完就停（它學會了在 `<|endoftext|>` 前結束）。
- **答案的範圍學到了**。`who` 的 221 個回覆全都是題目裡出現的三個人名之一，`color` 的 199 個全都是題目裡的兩個顏色之一，加、減、數數量的回覆全是數字。
- **加、減、數數量全對**（188／188、200／200、192／192）。這三種題型的答案是題目裡幾個字的函數，base model 在預訓練時早就讀過這些字，SFT 要做的是教它「現在要把答案寫出來」。
- **`color` 與 `who` 是亂猜**。`color` 在兩個顏色裡選，答對 101／199 = 51%；`who` 在三個人裡選，答對 74／221 = 33%，剛好是 1/2 與 1/3。`who` 答錯的 147 題，答的是第 1、2、3 個人的各是 48、42、57 題，平均分散。它知道答案在題目裡，但沒學會「找到被問的那樣東西，看是誰的」。

SFT 做的是**把模型已經有的能力引出來**，教它格式；它不保證憑空長出新能力。要做出查找這種能力，這個預算裡的 SFT 不夠。我們試了加料：

| 設定 | 合計 | `color`（199 題） | `who`（221 題） |
| --- | --- | --- | --- |
| 預設（seed 0） | 755／1000 | 101 | 74 |
| 預設，`--seed 1`（資料也不同） | 754／1000 | 94 | 80 |
| 4000 步、10 萬題、`lr_max=3e-4`（訓練迴圈 258 秒） | 753／1000 | 105 | 68 |

```bash
uv run tools/sft.py --out_dir runs/sft_seed1 --seed 1
uv run tools/sft.py --out_dir runs/sft_big --n_train 100000 --max_steps 4000 --lr_max 3e-4 --lr_min 3e-5 --eval_every 1000 --log_every 1000
```

訓練量變成 20 倍，兩種題型還是停在亂猜（`color` 94～105，`who` 68～80；亂猜的期望是 99.5 與 73.7）。為什麼學不起來，我們沒有進一步查，只知道在這個預算裡增加資料與步數沒有用。

這個結果對第 15 章其實是好事：加、減、數數量已經滿分，沒有進步的空間；`color` 與 `who` 卻留著大塊空間，答對的機率有 1/2 與 1/3，而且答對與答錯可以自動判斷，正是 RL 要處理的情況。

### 不套 mask 會怎樣

`SFTConfig(mask=False)` 把每條序列的 mask 全部換成 `True`：整串（題目、`Assistant:`、回覆）都算 loss。`val_loss` 與 `val_accuracy` 仍然只算回覆，數字才能比：

```bash
uv run tools/sft.py --out_dir runs/sft_nomask --no-mask
```

| 設定 | 250 步 `val_loss` | 1000 步 `val_loss` | 1000 題合計 |
| --- | --- | --- | --- |
| 套 mask（預設） | 0.310 | 0.220 | 755／1000 |
| 不套 mask | 0.381 | 0.217 | 737／1000 |

結果出乎意料：差別不大。不套 mask 的 `val_loss` 一開始慢一些（250 步 0.381 對 0.310），1000 步追平；最後的準確率差 18 題（1.8 個百分點），兩個種子的預設設定只差 1 題（755 與 754），差 18 題比這個大，但我們只跑了一次，不敢說是 mask 的功勞。

有兩個地方看得出兩種設定不一樣。一是訓練 loss：300 步之後，不套 mask 時在 0.43～0.50，因為提示裡的名字與東西是隨機挑的，模型永遠猜不準，這部分的 loss 壓不下去；套 mask 時是 0.07～0.29。二是 `grad_norm`：套 mask 時是 1.0～4.9，記錄到的每一筆都超過 `grad_clip=1.0`，會被裁剪；不套 mask 時是 0.5～1.3，只有一筆超過。我們推測是因為 loss 除以的 token 數不同：回覆只有 2 個 token，一批只有幾十個 token 在算，梯度的雜訊比一批近千個 token 大，這個推測沒有另外驗證。

這個玩具上 mask 沒有決定性，因為題目是模板填空，模型幾百步就學會模板；真實的 SFT 資料提示與回覆都很長，不套 mask 的差別我們沒有量過，不能從這裡推論。我們保留 mask 的理由是原則上的：提示是使用者給的、推論時不需要模型寫，不該花力氣去學。

### 回頭看評審

第 12 章第 4 集留了一個問題：標準模型當評審，永遠選同一個標籤。SFT 之後呢？我們把那集的評審提示放進 chat template（使用者說那段話，助理接著寫 ` A` 或 ` B`），base model 與 SFT 模型各測一次：

```python,norun
import random

import torch

from llmz.evaluation import choice_logprobs, judge_prompt, load_mcq, pick_choice
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import chat_pieces
from llmz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("data/tokenizer.json")
rng = random.Random(0)  # 和第 12 章第 4 集一樣：每題正確的結尾對一個隨機挑的錯誤結尾
pairs = []
for it in load_mcq("test"):
    wrong = rng.choice([c for i, c in enumerate(it["choices"]) if i != it["answer"]])
    pairs.append((it["context"], it["choices"][it["answer"]], wrong))


def judge_in_chat(model, context, first, second, labels):
    question = judge_prompt(context, first, second, labels).removesuffix("\n\nAnswer:")  # 結尾的 Answer: 換成 template 的 Assistant:
    prompt = "".join(text for text, _ in chat_pieces([{"role": "user", "content": question}], generation_prompt=True))
    logprobs, _ = choice_logprobs(model, tokenizer, prompt, [" " + label for label in labels])
    return pick_choice(logprobs)


for path in ["runs/base/model.pt", "runs/sft/model.pt"]:
    model = TransformerLM(ModelConfig())
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    for labels in [("A", "B"), ("B", "A"), ("1", "2")]:
        first_picked = right = consistent = 0
        for context, good, bad in pairs:
            good_first = judge_in_chat(model, context, good, bad, labels)  # 好的排前面：選 0 才對
            good_second = judge_in_chat(model, context, bad, good, labels)  # 好的排後面：選 1 才對
            first_picked += (good_first == 0) + (good_second == 0)
            right += (good_first == 0) + (good_second == 1)
            consistent += (good_first == 0) == (good_second == 1)  # 兩個順序選到同一個結尾
        print(f"{path} 標籤 {labels[0]}／{labels[1]}：選了排第一的 {first_picked}/200，答對 {right}/200，兩個順序一致 {consistent}/100")
```

輸出（沒有隨機性，CPU）：

```text
runs/base/model.pt 標籤 A／B：選了排第一的 200/200，答對 100/200，兩個順序一致 0/100
runs/base/model.pt 標籤 B／A：選了排第一的 0/200，答對 100/200，兩個順序一致 0/100
runs/base/model.pt 標籤 1／2：選了排第一的 3/200，答對 99/200，兩個順序一致 3/100
runs/sft/model.pt 標籤 A／B：選了排第一的 133/200，答對 95/200，兩個順序一致 15/100
runs/sft/model.pt 標籤 B／A：選了排第一的 60/200，答對 104/200，兩個順序一致 14/100
runs/sft/model.pt 標籤 1／2：選了排第一的 6/200，答對 100/200，兩個順序一致 2/100
```

base model 和第 12 章一樣：`A`／`B` 永遠選第一個，`B`／`A` 永遠選第二個，答對 100/200，兩個順序一致 0/100。SFT 之後不再每個標籤都一面倒（`A`／`B` 選第一個 133 次，`B`／`A` 只有 60 次；`1`／`2` 還是幾乎都選第二個），但答對還是在 200 題的一半上下（95～104），兩個順序一致的只有 15、14、2 題：選什麼仍然看標籤與位置，不看內容。

這不意外：SFT 的題目只有加、減、數數量、顏色、誰有什麼五種，從來沒有「比較兩個結尾」這種題。SFT 教的是它看過的那幾種指令的格式；沒看過的指令，它只能照這個格式亂答。要讓它學會評審，得有評審形式的 SFT 資料；這不是這個玩具的目標，我們沒有試。

### 存下來的 checkpoint

`runs/sft/model.pt` 是第 14 章後面兩集（偏好學習）與第 15 章（RL）的起點：

```python,norun
import torch

from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import make_tasks, reward, task_messages, generate_replies
from llmz.tokenizer import Tokenizer

model = TransformerLM(ModelConfig())
model.load_state_dict(torch.load("runs/sft/model.pt", map_location="cpu"))
tokenizer = Tokenizer.load("data/tokenizer.json")
task = make_tasks(1, seed=1, split="valid", kinds=("color",))[0]
print(task.question, generate_replies(model, tokenizer, [task_messages(task)]), task.answer)
```

後面要做的事都從這裡開始：偏好學習用「好回答對壞回答」繼續調整它；RL 用 `reward` 讓它答對 `color` 與 `who` 的機率變高。

## 重點整理

- SFT 迴圈與預訓練相同，差在：從標準模型載入、資料是 `(ids, mask)`、loss 是 `sft_loss`、學習率小十倍（`1e-4`）。`SFTConfig`、`train_sft`、`tools/sft.py` 與 smoke 設定（CPU 幾秒，`tests/test_posttrain.py`）都在 `llmz.posttrain`。
- M4 Max 上預設設定（1000 步、2 萬題）約 1 分鐘，存成 `runs/sft/model.pt`。1000 道 `"valid"` 題：base model 0/1000，SFT 755/1000（另一個種子 754、不套 mask 737、20 倍訓練量 753）。
- 學到格式（一個字、寫完就停）與答案的範圍；加、減、數數量全對；`color`（約 1/2）與 `who`（約 1/3）停在亂猜，增加資料與步數也沒用。SFT 引出模型已有的能力，不保證長出新能力。
- 不套 mask 在這個玩具上差別不大（250 步時 `val_loss` 慢一點，最後差 18 題）；原則上提示不該算 loss。
- 把第 12 章的評審題放進 chat template，SFT 後的模型仍然答不出（95～104/200）：沒教過的指令不會因為 SFT 而會。
