# DPO 推導與玩具版實作

## 本集目標

看懂 DPO（Direct Preference Optimization）怎麼從 RLHF 的目標推導出來，實作它的 loss，然後在 SFT 模型上用程式產生的偏好資料跑一次，量出它偏好正確回答的比例怎麼變，並看 \\( \beta \\) 的作用。

## 正文

### RLHF 要解的問題

上一集的流程叫 **RLHF**（reinforcement learning from human feedback，從人類回饋做強化學習）：先用偏好資料訓練一個獎勵模型 \\( r(x, y) \\)，再讓語言模型 \\( \pi \\) 去寫分數高的回答。只追分數會出問題：獎勵模型是從有限的偏好資料學來的，不完美，模型很快會找到它打分的漏洞，寫出分數很高、人看了卻很奇怪的回答。所以目標裡加一個拉回來的力量：

\\[ \max_\pi \; \mathbb{E}_{y \sim \pi(\cdot|x)}\big[ r(x, y) \big] - \beta \, \mathrm{KL}\big( \pi(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x) \big) \tag{1} \\]

- \\( \pi_{\text{ref}} \\) 是 **reference 模型**，這裡就是 SFT 之後、還沒動過的模型；\\( \pi \\) 是正在調整的 **policy**，從同一個起點開始。
- \\( \mathrm{KL}(\pi \| \pi_{\text{ref}}) = \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} \\) 衡量 \\( \pi \\) 離 \\( \pi_{\text{ref}} \\) 有多遠，相同時是 0。
- \\( \beta \\) 控制拉回來的力量：\\( \beta \\) 大，\\( \pi \\) 不敢離 reference 太遠；\\( \beta \\) 小，更敢追分數。

傳統做法用強化學習（第 15 章會正式介紹）去最佳化 (1)：要訓練獎勵模型，還要邊訓練邊讓模型自己生成回答、打分數。DPO（Rafailov 等人，2023）的洞察是：(1) 有一個可以直接寫出來的最佳解，反過來用它，可以不要獎勵模型，也不要生成回答，直接用偏好資料訓練 \\( \pi \\)。

### 最佳解長什麼樣子

(1) 的最佳解是：

\\[ \pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \, \exp\!\Big( \frac{r(x, y)}{\beta} \Big) \tag{2} \\]

reference 原本給的機率，乘上 \\( e^{r/\beta} \\)：獎勵高的回答被放大，低的縮小；\\( Z(x) \\) 是讓機率加起來等於 1 的常數。用四個可能的回覆驗證：

```python
import torch

from llmz.functional import softmax

beta = 0.5
ref = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)  # 四個可能的回覆，reference 給的機率
r = torch.tensor([1.0, 0.0, 2.0, -1.0], dtype=torch.float64)  # 每個回覆的獎勵


def objective(pi):
    """E[r] - beta * KL(pi || ref)：RLHF 要最大化的東西"""
    return (pi * r).sum() - beta * (pi * (pi / ref).log()).sum()


star = ref * torch.exp(r / beta)
Z = star.sum()
star = star / Z  # 公式 (2)：pi* = ref * exp(r / beta) / Z
print("pi* =", [round(p, 3) for p in star.tolist()])

greedy = torch.tensor([0.01, 0.01, 0.97, 0.01], dtype=torch.float64)  # 幾乎全押在獎勵最高的那個回覆
for name, pi in [("pi_ref（沒動）", ref), ("幾乎全押最高獎勵", greedy), ("pi*", star)]:
    print(f"{name}：目標值 {objective(pi).item():.4f}")

torch.manual_seed(0)
nearby = softmax(star.log() + 0.1 * torch.randn(10000, 4, dtype=torch.float64))  # 在 pi* 附近隨機挑一萬個機率分布
print("附近一萬個分布都沒有比 pi* 高：", all(objective(p) < objective(star) for p in nearby))
print("pi* 的目標值剛好是 beta * log Z：", abs(objective(star).item() - beta * Z.log().item()) < 1e-9)

implicit = beta * (star / ref).log()  # beta * log(pi* / ref)
print("beta * log(pi*/ref) 減掉 r：", [round(x, 4) for x in (implicit - r).tolist()])
print("-beta * log Z =", round(-beta * Z.log().item(), 4))
```

執行結果：

```text
pi* = [0.208, 0.021, 0.77, 0.001]
pi_ref（沒動）：目標值 0.7000
幾乎全押最高獎勵：目標值 1.2212
pi*：目標值 1.3262
附近一萬個分布都沒有比 pi* 高： True
pi* 的目標值剛好是 beta * log Z： True
beta * log(pi*/ref) 減掉 r： [-1.3262, -1.3262, -1.3262, -1.3262]
-beta * log Z = -1.3262
```

- \\( \pi^* \\) 的目標值最高，比「幾乎全押最高獎勵」還高：全押的 KL 太大。
- 附近一萬個隨機擾動的分布，沒有一個比它高。
- 最後兩行就是下一步：把 (2) 取 log 解出 \\( r \\)，\\( r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x) \\)。程式驗證了每個回覆的 \\( \beta \log(\pi^*/\pi_{\text{ref}}) - r \\) 都等於同一個數字 \\( -\beta \log Z \\)。

### 代進 Bradley–Terry

獎勵可以用 policy 與 reference 的機率比表示：\\( r = \beta \log(\pi/\pi_{\text{ref}}) + \beta \log Z \\)。把它代進上一集的 Bradley–Terry loss，\\( r_{\text{chosen}} - r_{\text{rejected}} \\) 裡兩個 \\( \beta \log Z(x) \\) 因為提示 \\( x \\) 相同，剛好抵消（上一集看過：分數整個平移，loss 不變）。\\( Z(x) \\) 不好算，現在不用算了：

\\[ \mathcal{L}_{\text{DPO}} = -\log \sigma\Big( \beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \Big) \tag{3} \\]

\\( y_w \\) 是 chosen，\\( y_l \\) 是 rejected。**policy 自己就是獎勵模型**：\\( \beta \log(\pi/\pi_{\text{ref}}) \\) 叫做**隱含獎勵**（implicit reward）。要訓練的東西只有一個語言模型，需要的資料是四個 log 機率：policy 與 reference 各算 chosen 與 rejected 一次。

### dpo_loss

```python,ignore
{{#include ../../../src/llmz/posttrain.py:dpo_loss}}
```

四個輸入都是整段回覆的 log 機率（`pair_logprobs` 的結果，`(B,)`）。它只是把 (3) 裡的兩個隱含獎勵丟進上一集的 `bradley_terry_loss`。

```python
import math

import torch

from llmz.posttrain import dpo_loss

ref_chosen, ref_rejected = torch.tensor([-3.0]), torch.tensor([-4.0])  # reference 給兩個回覆的 log 機率
print("log 2 =", round(math.log(2), 4))
for beta in [0.1, 1.0]:
    unmoved = dpo_loss(ref_chosen, ref_rejected, ref_chosen, ref_rejected, beta)
    print(f"beta {beta}：policy 還沒動 loss {unmoved.item():.4f}")

same = torch.tensor([-1.0])  # chosen 與 rejected 是同一個回覆
print("chosen 與 rejected 相同：", round(dpo_loss(same, same, ref_chosen, ref_chosen, 0.1).item(), 4))

# policy 把 chosen 抬高 1、把 rejected 壓低 1：隱含獎勵的差是 beta * 2
for beta in [0.1, 0.5, 1.0]:
    moved = dpo_loss(ref_chosen + 1, ref_rejected - 1, ref_chosen, ref_rejected, beta)
    print(f"beta {beta}：chosen +1、rejected -1 之後 loss {moved.item():.4f}")
```

執行結果：

```text
log 2 = 0.6931
beta 0.1：policy 還沒動 loss 0.6931
beta 1.0：policy 還沒動 loss 0.6931
chosen 與 rejected 相同： 0.6931
beta 0.1：chosen +1、rejected -1 之後 loss 0.5981
beta 0.5：chosen +1、rejected -1 之後 loss 0.3133
beta 1.0：chosen +1、rejected -1 之後 loss 0.1269
```

policy 還沒離開 reference 的時候，兩個 log 比值都是 0，loss 是 \\( \log 2 \\)，不管 \\( \beta \\) 是多少，也不管回覆本身的 log 機率是多少；`chosen` 與 `rejected` 是同一個回覆時，也是 \\( \log 2 \\)。梯度的方向由 `tests/test_posttrain.py` 檢查：`policy_chosen` 往上推，`policy_rejected` 往下推，起點的力道各是 \\( \beta/2 \\)。

### 訓練迴圈

```python,ignore
{{#include ../../../src/llmz/posttrain.py:train_dpo}}
```

和第 3 集的 `train_sft` 同一個骨架，差在：

- **兩個模型**：`model` 是 policy，會被更新；`reference` 從同一個檔案載入，之後不動，只做 forward（`torch.no_grad()`），不累積梯度也不被 optimizer 更新。兩者每一步看同一批 `ids`、`mask`。
- **資料**：`make_preferences` 的偏好對，預設只放 `color` 與 `who`（SFT 做不好的兩種）。
- **學習率**：`5e-5` 起跳，比 SFT 的 `1e-4` 小一半。DPO 沒有「標準答案」把模型拉住，每一步都只是在調整兩個回覆的機率差；學習率大一倍，後面會看到回答整個壞掉。
- **記錄**：`chosen_gain` 與 `rejected_gain` 是 policy 相對 reference 多給（或少給）`chosen`、`rejected` 的 log 機率；兩者的差乘上 \\( \beta \\) 就是 loss 裡的 margin。`val_preference` 是 `preference_rate`，`val_accuracy` 是 greedy 的答對比例。

### smoke 設定

和第 3 集一樣：byte-level tokenizer、隨機初始化的小模型（`sft_path` 是空字串，policy 與 reference 都從同一組隨機參數開始）。只用 16 對偏好資料、40 步，看迴圈有沒有照預期動：

```python
from llmz.model import ModelConfig
from llmz.posttrain import DPOConfig, train_dpo
from llmz.tokenizer import Tokenizer, base_vocab

END = "<|endoftext|>"
tokenizer = Tokenizer(base_vocab([END]), [], [END])
cfg = DPOConfig(
    sft_path="", out_dir="smoke_dpo",
    model=ModelConfig(vocab_size=257, context_length=256, d_model=64, n_layers=2, n_heads=4, d_ff=128),
    n_train=16, n_valid=20, batch_size=16, max_steps=40, lr_max=1e-3, lr_min=1e-4, warmup_steps=2, beta=0.1,
    log_every=10, eval_every=40, device="cpu",
)
records = train_dpo(cfg, tokenizer)
last = records[-1]
print("loss 降到 0.3 以下：", last["loss"] < 0.3)
print("chosen 被抬高、rejected 被壓低：", last["chosen_gain"] > 0 > last["rejected_gain"])
```

執行結果（某一次）：

```text
{'step': 10, 'loss': 0.3584, 'lr': 0.0009051132292283772, 'grad_norm': 0.235, 'time': 2.5, 'chosen_gain': 3.26, 'rejected_gain': -5.693, 'train_preference': 1.0}
{'step': 20, 'loss': 0.2039, 'lr': 0.0005871607054625496, 'grad_norm': 0.174, 'time': 4.8, 'chosen_gain': 6.859, 'rejected_gain': -8.74, 'train_preference': 1.0}
{'step': 30, 'loss': 0.1407, 'lr': 0.00024522329276841653, 'grad_norm': 0.145, 'time': 7.1, 'chosen_gain': 8.885, 'rejected_gain': -10.644, 'train_preference': 1.0}
{'step': 40, 'loss': 0.1201, 'lr': 0.0001, 'grad_norm': 0.132, 'time': 9.5, 'chosen_gain': 9.769, 'rejected_gain': -11.39, 'train_preference': 1.0, 'val_preference': 0.65, 'val_accuracy': 0.0}
loss 降到 0.3 以下： True
chosen 被抬高、rejected 被壓低： True
```

第一步的 loss 是 \\( \log 2 \\)（policy 就是 reference，`tests/test_posttrain.py` 檢查），之後降下來。`chosen_gain` 變正、`rejected_gain` 變負：policy 相對 reference 抬高了 `chosen`、壓低了 `rejected`。16 對題目反覆看，背下來了；這個設定只確認迴圈沒接錯，`val_preference`、`val_accuracy` 沒有意義（隨機初始化的小模型沒有學到 `color` 與 `who` 的規則）。

### 完整訓練

```bash
uv run tools/dpo.py
```

預設：偏好資料只放 SFT 做不好的 `color` 與 `who`（`kinds="color,who"`）、`n_train=20000` 對、`batch_size=32`、1000 步、學習率 `5e-5`（降到 `5e-6`）、`beta=0.1`。M4 Max 上跑兩次（種子 0 與 1），訓練迴圈各 107、112 秒（含 4 次驗證；較小的晶片會更慢）。存在 `runs/dpo/`，用法和 `runs/sft/model.pt` 完全一樣。種子 0 那次的紀錄（單次）。`val_preference` 與 `val_accuracy` 是 200 對／200 題 `color` 與 `who` 的驗證題：

| 步數 | 訓練 loss（單一 batch） | `val_preference` | `val_accuracy` |
| --- | --- | --- | --- |
| 250 | 0.706 | 0.545 | 0.470 |
| 500 | 0.632 | 0.645 | 0.580 |
| 750 | 0.418 | 0.870 | 0.815 |
| 1000 | 0.354 | 0.925 | 0.850 |

前 400 步 loss 在 0.66～0.72 之間，也就是 \\( \log 2 = 0.693 \\) 附近：沒學到東西；之後才開始下降。第 3 集的 SFT 在這兩種題型上停在亂猜，所以不能期待一開始就有動靜。

### 前後的偏好比例與正確率

用 1000 道 `"valid"` 題（和第 3、4 集同一批），量 SFT 模型與 DPO 後的模型：

```python,norun
import torch

from llmz.device import get_device
from llmz.model import ModelConfig, TransformerLM
from llmz.posttrain import KINDS, generate_replies, make_preferences, make_tasks, pair_batch, pair_logprobs, preference_rate, reward, task_messages, wrong_answers
from llmz.tokenizer import Tokenizer

device = get_device()
tokenizer = Tokenizer.load("data/tokenizer.json")
pairs = make_preferences(1000, seed=1, split="valid")
tasks = make_tasks(1000, seed=1, split="valid")


@torch.no_grad()
def mean_logprobs(model, some_pairs, chunk=50):
    """chosen 與 rejected 回覆的平均 log 機率（整段回覆的總和）"""
    chosen = rejected = 0.0
    for start in range(0, len(some_pairs), chunk):
        c, r = pair_logprobs(model, *[t.to(device) for t in pair_batch(tokenizer, some_pairs[start : start + chunk])])
        chosen, rejected = chosen + c.sum().item(), rejected + r.sum().item()
    return chosen / len(some_pairs), rejected / len(some_pairs)


for path in ["runs/sft/model.pt", "runs/dpo/model.pt"]:
    model = TransformerLM(ModelConfig()).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    replies = generate_replies(model, tokenizer, [task_messages(t) for t in tasks])
    print(path)
    row_rate, row_hits = [], []
    for kind in KINDS:
        rate = preference_rate(model, tokenizer, [p for p in pairs if p.kind == kind])
        hits = [reward(t, r) for t, r in zip(tasks, replies) if t.kind == kind]
        row_rate.append(f"{kind} {rate:.1%}")
        row_hits.append(f"{kind} {int(sum(hits))}/{len(hits)}")
    print("  偏好比例：", "  ".join(row_rate))
    print("  答對：", "  ".join(row_hits), f"  合計 {sum(reward(t, r) for t, r in zip(tasks, replies)):.0f}/1000")
    chosen, rejected = mean_logprobs(model, [p for p in pairs if p.kind in ("color", "who")])
    inside = sum(r.strip() in [t.answer, *wrong_answers(t)] for t, r in zip(tasks, replies))
    print(f"  color 與 who 的平均 log 機率：chosen {chosen:.2f}　rejected {rejected:.2f}")
    print(f"  回覆落在題目的候選裡：{inside}/1000")
```

M4 Max 上的輸出（MPS；greedy 沒有隨機性，但每次訓練出來的 `model.pt` 略有不同）：

```text
runs/sft/model.pt
  偏好比例： add 100.0%  subtract 100.0%  count 100.0%  color 50.8%  who 46.2%
  答對： add 188/188  subtract 200/200  count 192/192  color 101/199  who 74/221   合計 755/1000
  color 與 who 的平均 log 機率：chosen -1.03　rejected -0.99
  回覆落在題目的候選裡：1000/1000
runs/dpo/model.pt
  偏好比例： add 100.0%  subtract 98.5%  count 93.2%  color 99.5%  who 88.2%
  答對： add 188/188  subtract 183/200  count 109/192  color 198/199  who 155/221   合計 833/1000
  color 與 who 的平均 log 機率：chosen -1.07　rejected -11.57
  回覆落在題目的候選裡：959/1000
```

- **偏好比例**：`color` 50.8% → 99.5%，`who` 46.2% → 88.2%。SFT 時分不出來的兩種題型，現在給正確答案的 log 機率幾乎都比較高。
- **答對**：`color` 101 → 198／199，`who` 74 → 155／221。第 3 集用 20 倍的資料與步數做 SFT 都停在亂猜的題型，1000 步 DPO 就學會了「看題目找被問的那一個」。我們沒有查為什麼 DPO 學得起來、SFT 學不起來；一個沒驗證的猜測是：偏好對把同一題的正確與錯誤名字並排，梯度直接指向「這兩個的差別在哪」，而 SFT 只告訴模型「寫這個」。
- **副作用**：偏好資料裡沒有的題型變差了。`count` 192／192 → 109／192、`subtract` 200／200 → 183／200，而且有 41 題的回覆不在題目的候選裡。DPO 沒碰過這些題，但 policy 的參數整個被推動了。
- **`chosen` 沒有變高，`rejected` 掉很多**：`color` 與 `who` 的 `chosen` 平均 log 機率 \\( -1.03 \\) → \\( -1.07 \\)，`rejected` \\( -0.99 \\) → \\( -11.57 \\)。DPO 的 loss 只看兩者的**差**，所以「把 `rejected` 壓下去」就能降低它，`chosen` 不必變高。

第二次執行（`--seed 1`，偏好資料與 batch 順序都不同）：合計 850／1000，`color` 196、`who` 162、`count` 133、`subtract` 171，`chosen` \\( -0.87 \\)、`rejected` \\( -9.74 \\)，回覆在候選裡的有 981 題。兩次的範圍：合計 833～850，`color` 196～198，`who` 155～162，`count` 109～133。

```bash
uv run tools/dpo.py --out_dir runs/dpo_seed1 --seed 1
```

### β 的作用

\\( \beta \\) 是 (1) 裡那條把 policy 拉回 reference 的繩子：\\( \beta \\) 大，繩子緊，policy 離 reference 不遠；\\( \beta \\) 小，policy 可以走很遠。我們換 \\( \beta \\)，其他設定不變，每個值各跑種子 0 與 1（\\( \beta = 5 \\) 只跑種子 0），同樣用上面的腳本量：

```bash
uv run tools/dpo.py --out_dir runs/dpo_beta1 --beta 1.0
uv run tools/dpo.py --out_dir runs/dpo_beta1_seed1 --beta 1.0 --seed 1
```

其他的 \\( \beta \\) 同樣換 `--beta`、`--seed` 與 `--out_dir`。

| \\( \beta \\) | 種子 | 合計 | `subtract` | `count` | `color` | `who` | `chosen`／`rejected` 平均 log 機率 | 回覆在候選裡 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT（起點） | | 755 | 200 | 192 | 101 | 74 | −1.03／−0.99 | 1000 |
| 0.01 | 0 | 846 | 164 | 127 | 197 | 171 | −1.15／−15.55 | 956 |
| 0.01 | 1 | 598 | 186 | 93 | 105 | 27 | −5.71／−9.94 | 758 |
| 0.1 | 0 | 833 | 183 | 109 | 198 | 155 | −1.07／−11.57 | 959 |
| 0.1 | 1 | 850 | 171 | 133 | 196 | 162 | −0.87／−9.74 | 981 |
| 1 | 0 | 901 | 196 | 169 | 196 | 152 | −0.46／−2.90 | 1000 |
| 1 | 1 | 895 | 197 | 166 | 197 | 147 | −0.44／−3.22 | 1000 |
| 5 | 0 | 776 | 200 | 192 | 104 | 92 | −1.11／−1.16 | 1000 |

（`add` 全是 187～188／188，沒有列。「合計」與各題型都是答對的題數，1000 題裡 `subtract` 200、`count` 192、`color` 199、`who` 221 題。）

- **\\( \beta = 5 \\)：繩子太緊，幾乎沒動**。偏好比例 52.3%／53.8%，`chosen` 與 `rejected` 的 log 機率和 SFT 差不多。`color` 與 `who` 之外的題型和 SFT 一樣好（`subtract` 200、`count` 192），但這兩種題型也沒學到（104、92）。
- **\\( \beta = 1 \\)：這組設定裡最好的**。`color` 與 `who` 學到了（196～197、147～152），別的題型保住了大半（`count` 166～169、`subtract` 196～197），`rejected` 只掉到約 \\( -3 \\)，1000 題的回覆都還在候選裡。兩次的合計 895～901。
- **\\( \beta = 0.01 \\)：繩子太鬆，不穩定**。一次學得很好（846，`who` 171 是全表最高），一次完全沒學起來（598，`color` 105、`who` 27、`chosen` 掉到 \\( -5.71 \\)）。兩次差這麼多，單次結果不可信。
- \\( \beta = 0.1 \\)（預設）在中間，兩次的合計 833 與 850。

為什麼 \\( \beta \\) 大就停得早？練習題算過：隱含獎勵的差是 \\( \beta \\) 乘上 log 比值的差，\\( \beta \\) 大時 sigmoid 很快飽和，這一對「已經做對了」，梯度趨近 0。\\( \beta \\) 小時要把 `rejected` 壓得很低，loss 才降得下來。這個表是同一份資料、同一個學習率下的結果；學習率與 \\( \beta \\) 會互相影響，我們沒有掃二維的格子。

### 偏好比例 100% 不等於回答變好

學習率調大一倍、步數變成三倍：

```bash
uv run tools/dpo.py --out_dir runs/dpo_lr1e-4 --lr_max 1e-4 --lr_min 1e-5 --max_steps 3000
```

`color` 與 `who` 的偏好比例是 100.0%／100.0%，但 `chosen` 的平均 log 機率是 \\( -12.18 \\)、`rejected` 是 \\( -82.51 \\)，greedy 的回覆只有 199／1000 落在候選裡，合計答對 199／1000（`color` 199／199，其他題型全部 0）。兩個回答的差拉得很開，但兩個都已經不是模型會寫的東西了。**偏好比例只比較同一題的兩個回答，沒有檢查模型自己寫出來的是什麼**；訓練時要同時看 `chosen_gain`（掉太多就是警訊）和生成出來的回答。

另一個我們試過、沒有幫助的做法：把五種題型都放進偏好資料（`uv run tools/dpo.py --out_dir runs/dpo_all5 --kinds add,subtract,count,color,who`，其他設定不變）。1000 步之後 `color` 與 `who` 的偏好比例只有 53.3%／50.2%，合計 610／1000，比 SFT 還差。我們只跑了一次，沒有查原因。

## 重點整理

- RLHF 的目標是「獎勵的期望，減 \\( \beta \\) 倍 policy 與 reference 的 KL」。它的最佳解是 \\( \pi^* \propto \pi_{\text{ref}} e^{r/\beta} \\)；反過來，獎勵可以用 \\( \beta \log(\pi/\pi_{\text{ref}}) \\) 表示，代進 Bradley–Terry，\\( Z(x) \\) 抵消，得到 DPO loss：只需要 policy 與 reference 對 `chosen`、`rejected` 的 log 機率，不需要獎勵模型，也不需要邊訓練邊生成回答。
- `dpo_loss` 重用 `bradley_terry_loss`；policy 沒離開 reference 時是 \\( \log 2 \\)，梯度的方向是抬高 `chosen`、壓低 `rejected`。`DPOConfig`、`train_dpo`、`tools/dpo.py` 與 smoke 設定（CPU 幾秒，`tests/test_posttrain.py`）都在 `llmz.posttrain`。
- M4 Max 實測（預設設定，1000 步，約 110 秒）：`color` 的偏好比例 50.8% → 98.5～99.5%，答對 101 → 196～198／199；`who` 46.2% → 88.2～88.7%，答對 74 → 155～162／221。偏好資料裡沒有的題型變差（`count` 192 → 109～133）。
- \\( \beta \\) 是拉回 reference 的力量：\\( \beta = 5 \\) 幾乎沒動，\\( \beta = 1 \\) 這組設定裡最好（合計 895～901）、保住了其他題型，\\( \beta = 0.01 \\) 不穩定（598 與 846）。
- DPO 的 loss 只看 `chosen` 與 `rejected` 的差：`rejected` 可以掉很多、`chosen` 也可能跟著掉，學習率太大時偏好比例 100% 但回答全壞。要同時看 `chosen_gain` 與實際生成。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/posttrain.py` 的 `chat_pieces`、`encode_chat`、`pad_batch`、`sft_loss`、`make_tasks`（連同五個 `make_*` 題目產生器）、`task_messages`、`reward`、`generate_replies`、`accuracy`、`train_sft`、`bradley_terry_loss`、`dpo_loss`、`wrong_answers`、`make_preferences`、`pair_batch`、`pair_logprobs`、`preference_rate`、`train_dpo`。把它們的本體換成 `raise NotImplementedError`，執行：

```bash
uv run pytest tests/test_posttrain.py
```

自己寫回來讓測試通過。

再試一題：替 `DPOConfig` 加一個 `sft_weight`，把 `chosen` 的 `sft_loss` 乘上它，加進 `dpo_loss` 一起訓練。用 `--beta 0.1` 與 `--sft_weight 0` 和 `1` 各跑一次，比較 `chosen` 的平均 log 機率與 `count` 的答對題數，這一項能不能擋住 `chosen` 往下掉？再試一題：改寫 `preference_rate`，讓它額外回傳每一對的 margin（`chosen` 減 `rejected` 的 log 機率），找出 DPO 之後 `who` 仍然選錯的那些對，看看它們有什麼共同點。
