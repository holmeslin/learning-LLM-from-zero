# vLLM rollout（選讀：需 GPU）

## 本集目標

知道真實的 RL 訓練為什麼把 rollout 交給 vLLM 這類推論引擎，它省掉什麼、又多出哪些要處理的事。這一集的 vLLM 程式需要 NVIDIA 顯示卡與另外安裝的 vLLM（本書不把它列為相依），Mac 上只能讀；本書自己的程式量出來的數字，在 Mac 上跑得出來。

## 正文

### rollout 的時間花在哪裡

第 5 集量過：把 `generate_cached` 換成 `sample_batch`，128 條回答的 rollout 快了約 20 倍。那只是把取樣寫得更適合 RL。更根本的問題在一步 RLVR 的兩半，成本長得不一樣：

- **rollout**：每條回答要 decode 一個 token 一個 token 地生成（第 11 章第 2 集），回答多長就要跑多少步，一步接著一步，不能平行。
- **更新**：把整批提示加回答一次餵進模型，做 forward 與 backward，所有 token 同時算，一次就完。

所以回答變長時，兩半的成長速度不同。在 M4 Max 上，用第 5 集那個 `runs/sft_short` 模型，同樣 128 條回答，只改每條回答生成幾個 token，量 rollout 與更新（`old_logp` 一次、對同一批更新 2 次，含 `AdamW`）各要多久：

```python,norun
import torch

from llmz.device import get_device
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW, clip_grad_norm_
from llmz.posttrain import make_tasks
from llmz.profile import timeit
from llmz.rl import group_advantages, grpo_loss, rollout, token_logprobs
from llmz.tokenizer import Tokenizer

device = get_device()
tokenizer = Tokenizer.load("data/tokenizer.json")
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/sft_short/model.pt", map_location=device))
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
tasks = make_tasks(16, seed=0, kinds=("add", "subtract"))  # 16 題 * 8 個回答 = 128 條，和第 5 集一樣


def update(ids, mask, advantages):
    """第 5 集一步 RLVR 的後半：算一次 old_logp，再對同一批更新 2 次（forward、backward、裁剪、AdamW）。"""
    with torch.no_grad():
        old_logp = token_logprobs(model, ids)
    for _ in range(2):
        grpo_loss(token_logprobs(model, ids), old_logp, advantages, mask).backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()


print("每條回答的 token 數  rollout    更新       rollout 的比例")
for n in [4, 16, 64, 128, 192]:
    out = rollout(model, tokenizer, tasks, 8, n)
    ids, mask = out.ids.to(device), out.mask.to(device)[:, 1:]
    advantages = group_advantages(out.rewards).reshape(-1).to(device)
    t_rollout = timeit(lambda: rollout(model, tokenizer, tasks, 8, n), device, warmup=1, repeat=3)
    t_update = timeit(lambda: update(ids, mask, advantages), device, warmup=1, repeat=3)
    print(f"{n:>10d}  {t_rollout:8.2f} 秒  {t_update:7.2f} 秒  {t_rollout / (t_rollout + t_update):8.0%}")
```

M4 Max 上的輸出（某一次）：

```text
每條回答的 token 數  rollout    更新       rollout 的比例
         4      0.16 秒     0.24 秒       40%
        16      0.60 秒     0.38 秒       61%
        64      2.35 秒     1.29 秒       65%
       128      4.95 秒     2.32 秒       68%
       192      7.22 秒     3.28 秒       69%
```

同一段程式在同一台 M4 Max 上跑了三次。rollout 的秒數很穩（回答 4、16、64、128、192 個 token：0.15～0.19、0.57～0.60、2.35～2.38、4.64～5.03、6.82～7.38 秒），更新的秒數有一次明顯比另外兩次快（64 個 token：0.76～1.29 秒，128 個：1.46～2.32 秒），我們沒有查原因。所以我們只用範圍下結論：

- **rollout 的時間和回答長度成正比**。多一個 token 就多一步 decode，128 條一起算，一步約 0.036～0.038 秒（6.82～7.38 秒除以 192），不管前面已經生成多少。
- **更新也隨長度增加**（所有 token 還是都要算），但一開始慢得多：回答 4 個 token 時，rollout 只占一步的 39～44%；16 個 token 以上占六成以上；64 個 token 以後是 65～76%，不再明顯上升，因為更新的時間隨總 token 數也在成長。
- 所以在我們的設定裡，回答一長，rollout 就占掉一步的三分之二到四分之三。這是用只有 4 層的小模型量的；模型越大，decode 的每一步要讀的權重越多（第 11 章第 2 集），要等一條幾百到幾千個 token 的回答，rollout 只會更慢。

OpenRLHF 的 README 說，RLHF 訓練有 80% 的時間花在取樣生成上，它用 vLLM 來加速這一段。我們沒有量過他們的系統，這句話是他們的說法，和我們量到的七成同一個量級；它的原因就是上面的趨勢：回答越長（例如要一步一步推理的數學題，一條回答可能是幾百到幾千個 token），rollout 這一半就越重。

### 推論引擎：vLLM

第 11 章第 4 集講過推論伺服器要處理的三件事：長度不同、結束時間不同、cache 的空間。vLLM（Kwon 等人，2023，arXiv 2309.06180）就是為這些設計的推論引擎：

- **continuous batching**：序列一結束，就馬上換成排隊中的新請求，不用等整個 batch。`sample_batch` 的每條回答都生成滿 `max_new_tokens` 個 token，再用 `mask` 遮掉結尾之後的部分，結尾之後的步數是白算的。
- **PagedAttention**：KV cache 切成固定大小的小塊、用到才配置，不像我們的 `KVCache` 照最長長度預先配置。論文報告，同樣的延遲下，吞吐量比 FasterTransformer 與 Orca 高約 2～4 倍。
- **prefix caching**：RLVR 的同一題要抽 `group_size` 個回答，這 8 條回答的提示完全一樣。我們的 `rollout` 把提示重複 8 次放進 batch，prefill 做了 8 次；vLLM 的 automatic prefix caching 把相同前綴的 KV cache 存起來重用，prefill 只算一次。（它只省 prefill 的時間，decode 不受影響。）

用法長這樣：一題要 8 個回答，在 `SamplingParams` 裡寫 `n=8`，`temperature=1.0` 是照模型本來的機率抽（第 5 集說過為什麼要這樣）：

```python,norun
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")  # vLLM 文件範例用的模型；任何 vLLM 支援的 Hugging Face 模型都行
params = SamplingParams(n=8, temperature=1.0, max_tokens=4, logprobs=1)  # 每題抽 8 個回答，每個最多 4 個 token，並回傳 log 機率
outputs = llm.generate(["User: 2 + 3?\nAssistant:", "User: 7 - 4?\nAssistant:"], params)  # 兩題

for out in outputs:  # 每一題一個 RequestOutput
    print(out.prompt_token_ids)  # 提示的 token id
    for completion in out.outputs:  # 這一題的 8 個回答
        print(completion.token_ids, repr(completion.text), completion.finish_reason)
```

這段程式沒有輸出可以貼：我們沒有 NVIDIA 顯示卡，沒有跑過它，欄位的名稱照 vLLM 的文件（`RequestOutput` 的 `prompt_token_ids` 與 `outputs`，每個 `CompletionOutput` 的 `text`、`token_ids`、`logprobs`、`finish_reason`）。要接進我們的訓練迴圈，只要把每條回答整理成第 5 集 `Rollout` 的樣子：提示加回答的 `ids`、只涵蓋回答的 `mask`、`reward` 打出來的 `rewards`。結尾的 token 在不在 `token_ids` 裡，要看 vLLM 的版本與設定，接的時候自己檢查（第 5 集的 `mask` 把結尾算進回答）。

本書的 `TransformerLM` 結構和 Llama 系列很像（RMSNorm、SwiGLU、RoPE），但 vLLM 載入的是 Hugging Face 格式的模型：權重的名稱與形狀、RoPE 的實作細節、tokenizer 都要對上，才載得進去。本書不做這件事。

### 用了推論引擎之後，多出來的三件事

**權重要同步**。訓練的程式每一步更新權重，vLLM 手上的是另一份。下一次 rollout 之前，要把新權重送過去，否則取樣用的是舊的 policy（第 5 集的 `old_logp` 以為取樣的就是現在這一版）。權重越大，每一步搬的越多。Hugging Face 的 TRL 的 `GRPOConfig(use_vllm=True)` 有兩種模式：`colocate`（vLLM 與訓練在同一批顯示卡上）與 `server`（vLLM 另外跑在別的顯示卡上，訓練程式把請求與新權重送過去）；OpenRLHF、verl 也都是用 vLLM（verl 還可以用 SGLang）做 rollout。同步與排程的細節隨版本一直在變，這裡不寫。

**兩邊算出來的機率不完全一樣**。訓練程式與 vLLM 用不同的 kernel 與精度，同一份權重、同一條回答，算出來的 log 機率會差一點點。所以我們一直是在訓練程式裡重算 `old_logp`（第 4 集），不用引擎回傳的 `logprobs`：這樣第一次更新時比值恰好是 1。Zheng 等人在 GSPO 的論文裡也這樣說：因為訓練引擎與推論引擎（他們舉 Megatron、SGLang 與 vLLM）的精度有差，實務上通常用訓練引擎重算舊 policy 的機率；並且主張序列層級的比值（第 6 集）比 token 層級的更能容忍這種差距，可能可以直接用推論引擎回傳的機率，省掉這次重算。

**要等最慢的那一條**。同一批 128 條回答裡，有的短、有的寫滿上限，整批 rollout 要等最長的那條生成完，訓練的顯示卡這段時間在等。這也是為什麼 OpenRLHF 的 README 提到 partial rollout：讓生成與訓練重疊，代價是同一條回答可能前半用舊權重、後半用新權重；它們建議搭配 token 層級的重要性取樣校正。

### 我們的 `sample_batch` 與 vLLM 差在哪裡

| | `sample_batch` | vLLM |
| --- | --- | --- |
| 提示 | 長度相同的放同一個 batch | 長度不同的一起處理 |
| 結束 | 生成滿 `max_new_tokens`，事後用 `mask` 遮掉 | 結束的序列馬上換成新請求 |
| KV cache | 照最長長度預先配置 | 切成小塊，用到才配置 |
| 同一題的 G 個回答 | 提示重複 G 次，prefill G 次 | 可以重用相同前綴的 cache |
| 權重 | 就是訓練的那一份 | 另一份，每步要同步 |

最後一列是我們的版本唯一省事的地方：訓練與取樣是同一個模型，沒有同步、沒有機率的差距。這也是這本書的 RL 能在 M4 Max 上跑完的原因，代價是沒有上面的效率。

## 重點整理

- 一步 RLVR 的 rollout 是一個 token 接一個 token 地 decode，時間和回答長度成正比（M4 Max 上 128 條一起、一步約 0.036～0.038 秒）；更新是整批一次 forward／backward。回答 4 個 token 時 rollout 占一步的約 4 成，64 個 token 以上約 65～76%；OpenRLHF 的 README 說，他們的 RLHF 訓練有 80% 的時間在取樣。
- vLLM 用 continuous batching、PagedAttention 與 prefix caching 加速取樣；同一題的 `group_size` 個回答共用相同的提示，prefix caching 正好用得上。
- 用推論引擎要多處理三件事：每一步把新權重同步過去；用訓練程式重算 `old_logp`，因為兩邊的機率會有一點差距；整批要等最長的回答（partial rollout 是一種做法）。
- 我們的 `sample_batch` 是訓練與取樣共用同一個模型的簡化版，沒有同步的問題，也沒有上面這些效率。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/rl.py` 的 `sample_actions`、`bandit_rewards`、`expected_reward`、`reinforce_loss`、`estimate_gradient`、`train_bandit`、`group_advantages`、`verifiable_rewards`、`train_groups`、`token_logprobs`、`importance_ratio`、`grpo_loss`、`sample_batch`、`rollout`、`train_rlvr`。把它們的本體換成 `raise NotImplementedError`，執行：

```bash
uv run pytest tests/test_rl.py
```

自己寫回來讓測試通過。

再試一題：替 `grpo_loss` 加上原論文的 KL 懲罰項，`train_rlvr` 以訓練開始前的模型當 reference（第 5 集的結論裡，沒練的 `count` 退步，那裡沒有 KL）。在 `runs/sft_short` 上練 `add` 與 `subtract`，比較加了 KL 之後，`count` 的答對題數，與 `add`、`subtract` 的進步有什麼不同。再試一題：本章的結論是兩個種子分不出 GRPO 與 Dr. GRPO 的高下。再跑種子 2、3（`--seed 2`、`--seed 3`），看兩者的 1000 題正確率排序是否穩定，還是又隨種子對調。
