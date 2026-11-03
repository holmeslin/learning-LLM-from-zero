# perplexity 的限制

## 本集目標

看清楚 perplexity 在什麼情況下不能拿來比：換了 tokenizer、換了領域，或文字本身很單調。學會用 bits per byte 把不同 tokenizer 的結果換成同一個單位。

## 正文

### 回顧：perplexity 是什麼

第 1 章第 15 集說過，perplexity 是平均 cross-entropy 取 exp：模型在每個位置平均像是從幾個「機率一樣大」的 token 裡猜一個。第 5 章訓練時看的驗證集 loss，第 6、10 章比較消融與 scaling law，用的都是它。

它有兩個好處：算起來便宜（一次 forward），而且是平滑的數字，小小的進步也看得出來。這一集看它的三個陷阱。

### 陷阱一：單位是「每個 token」

perplexity 除的是 token 的個數，但不同的 tokenizer 會把同一段文字切成不同個數的 token。下面假設兩個模型給同一段 600 bytes 的文字一樣的總機率（總 loss 都是 360 nat，也就是一樣好），只是 tokenizer 不同：

```python
import math

total_nll = 360.0  # 模型給這段文字的總 -log 機率，單位 nat
n_bytes = 600
for name, n_tokens in [("切成 150 個 token", 150), ("切成 100 個 token", 100)]:
    print(f"{name}：每個 token 的 perplexity {math.exp(total_nll / n_tokens):.2f}，"
          f"{total_nll / n_bytes / math.log(2):.3f} bits/byte")
```

執行結果：

```text
切成 150 個 token：每個 token 的 perplexity 11.02，0.866 bits/byte
切成 100 個 token：每個 token 的 perplexity 36.60，0.866 bits/byte
```

總機率一樣，每個 token 的 perplexity 卻差了 3 倍多。切得越細，每個 token 越好猜，perplexity 就越低，但那不是模型比較好。

比較公平的單位是 **bits per byte**（每個 byte 要用幾個 bit 編碼）：把文字的總 \\( -\log p \\) 除以 byte 數，再除以 \\( \ln 2 \\) 換成 bit。byte 數與 tokenizer 無關，所以不同 tokenizer 的模型也比得起來。

用真的資料看看。拿標準語料的驗證集，比三個「模型」：第 5 章訓練好的 Transformer，以及兩個只看出現比例的 unigram 模型，一個把文字當成 token，一個當成 byte。unigram 模型沒有上下文，每個位置都用同一張機率表（訓練集裡每個符號出現的比例，每個次數先加 1 避免機率是 0）：

```python,norun
import math
import os

import numpy as np
import torch

from llmz.data import load_tokens
from llmz.device import get_device
from llmz.model import ModelConfig, TransformerLM
from llmz.train import TrainConfig, evaluate

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()

train, valid = load_tokens("data/train.bin"), load_tokens("data/valid.bin")
valid_bytes = os.path.getsize("data/valid.txt")  # valid.bin 就是 valid.txt 用標準 tokenizer 切出來的
bytes_per_token = valid_bytes / len(valid)

# 模型一：第 5 章的 Transformer，每個 token 的平均 loss（nat）
torch.manual_seed(0)
loss_model = evaluate(model, valid, TrainConfig(eval_batches=100), device)

# 模型二：token 的 unigram，只看每個 token 在訓練集出現的比例
counts = np.bincount(np.asarray(train), minlength=10000) + 1
loss_unigram = -np.log(counts / counts.sum())[np.asarray(valid)].mean()

# 模型三：byte 的 unigram，把文字當成一個一個 byte
byte_counts = np.bincount(np.frombuffer(open("data/train.txt", "rb").read(), dtype=np.uint8), minlength=256) + 1
valid_text = np.frombuffer(open("data/valid.txt", "rb").read(), dtype=np.uint8)
loss_byte = -np.log(byte_counts / byte_counts.sum())[valid_text].mean()

for name, unit, loss, bytes_per_unit in [
    ("byte unigram", "byte", loss_byte, 1.0),
    ("token unigram", "token", loss_unigram, bytes_per_token),
    ("Transformer", "token", loss_model, bytes_per_token),
]:
    print(f"{name}：每個 {unit} 的 perplexity {math.exp(loss):.1f}，{loss / bytes_per_unit / math.log(2):.2f} bits/byte")
```

M4 Max 上的輸出：

```text
byte unigram：每個 byte 的 perplexity 21.9，4.45 bits/byte
token unigram：每個 token 的 perplexity 332.6，2.03 bits/byte
Transformer：每個 token 的 perplexity 5.4，0.59 bits/byte
```

只看 perplexity 欄，byte unigram 的 21.9 比 token unigram 的 332.6 小了 15 倍，好像好很多。換成 bits/byte 結果就反過來：token unigram 每個 byte 只要 2.03 bit，byte unigram 要 4.45 bit，後者其實差了 2 倍多。byte 的 perplexity 小，只是因為一個 byte 比一個 token 好猜得多。標準模型是 0.59 bits/byte（標準語料平均每個 token 4.12 bytes，loss 約 1.69 nat），遠遠勝過兩個 unigram。

`evaluate` 是第 5 章的函式，隨機抽 100 批、每批 32 段、每段 256 個 token（共約 82 萬個 token），所以這裡的 loss 1.69 和訓練記錄裡只抽 20 批的 1.70 略有不同。

所以讀論文時要看清楚單位：不同 tokenizer 的模型要比較，得用 bits per byte（或 bits per character）這類與切法無關的數字。

### 陷阱二：領域不同，數字就不能比

同一個模型，在不同的文字上量 perplexity。下面五段都是為這一集原創的短文字，每段只有幾十個 token，只是示範用：

```python,norun
import math

import torch

from llmz.device import get_device
from llmz.functional import log_softmax
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")


@torch.no_grad()
def token_nll(ids: torch.Tensor) -> torch.Tensor:
    """ids (1, T) -> (T-1,)：每個 token 的 -log p(它 | 前面所有 token)。第一個 token 沒有前文，不算。"""
    logp = log_softmax(model(ids[:, :-1]), dim=-1)[0]  # (T-1, V)
    return -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]]  # 和第 1 章的 cross_entropy 一樣，每個位置只取正確答案那一格


texts = {
    "兒童故事": "Once upon a time, there was a little girl named Mia. She had a red ball. Every day she played with it in the park. One day, the ball rolled into a pond. Mia felt sad, but her dog jumped in and brought it back. Mia gave him a big hug.",
    "新聞": "The city council voted on Tuesday to approve a new budget for the coming fiscal year. The plan raises spending on road repairs by eight percent, while funding for the public library will stay the same. Several residents spoke against the proposal during the meeting.",
    "法律": "The licensee shall not, without the prior written consent of the licensor, assign or transfer any of its rights or obligations under this agreement, whether in whole or in part, to any third party.",
    "Python": "def add_all(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total\n\nprint(add_all([1, 2, 3]))",
    "中文": "從前有一個小女孩，她有一顆紅色的球。每天她都在公園裡玩球。有一天，球滾進了池塘裡，小女孩很傷心。",
}
for name, text in texts.items():
    ids = torch.tensor([tokenizer.encode(text)], device=device)
    nll = token_nll(ids)
    n_bytes = len(text.encode())
    print(f"{name}：{ids.shape[1]} 個 token，每個 token {n_bytes / ids.shape[1]:.2f} bytes，"
          f"perplexity {math.exp(nll.mean().item()):.1f}，{nll.sum().item() / n_bytes / math.log(2):.2f} bits/byte")
```

`token_nll` 是第 1 章 cross-entropy 的逐 token 版本：先 `log_softmax`，再用索引張量取出每個位置上「正確的下一個 token」的 log 機率，加負號，但不取平均。第 3 集會把它擴充成帶 mask 的版本。

M4 Max 上的輸出：

```text
兒童故事：60 個 token，每個 token 3.88 bytes，perplexity 5.7，0.64 bits/byte
新聞：72 個 token，每個 token 3.69 bytes，perplexity 1048.8，2.68 bits/byte
法律：52 個 token，每個 token 3.79 bytes，perplexity 1672.4，2.77 bits/byte
Python：62 個 token，每個 token 1.94 bytes，perplexity 13068.6，6.95 bits/byte
中文：144 個 token，每個 token 1.00 bytes，perplexity 426.0，8.67 bits/byte
```

同一個模型，perplexity 從 5.7 到 13068.6。模型沒有壞掉，它只讀過 TinyStories：沒見過新聞用語、法律條文與程式碼，中文則幾乎每個字都被切成 3 個單 byte 的 token（每個 token 1.00 bytes）。中文那一行的 8.67 bits/byte 甚至比「從 256 種 byte 裡亂猜」的 8 bits/byte 還差，因為模型把大部分機率都留給了英文的 token。

所以 perplexity 回答的是「這段文字離模型的訓練資料有多遠」，而不是「模型有多好」。要比較兩個模型，必須用**同一份**資料。論文裡報的 perplexity 數字，要看它是在哪個資料集上量的。

### 陷阱三：低 perplexity 不等於好文字

perplexity 衡量的是模型對一段文字「有多不意外」，不是這段文字有沒有用。我們讓模型自己寫故事，再用它自己算這些字的 perplexity：一種用 greedy，每一步選機率最大的 token；一種用 temperature 1.0 直接依機率抽。每種都只算模型自己寫的部分，不含 prompt。

```python,norun
import math

import torch

from llmz.device import get_device
from llmz.functional import log_softmax
from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")

prompts = [
    "Once upon a time, there was a little girl named Lily.",
    "The dog saw a big",
    "Tom and his mom went to the park.",
    "One day, a bird flew into the garden.",
]


@torch.no_grad()
def perplexity_after(ids: torch.Tensor, start: int) -> tuple[float, int]:
    """ids (1, T)：第 start 個 token 以後（也就是模型自己寫的部分）的 -log p 總和與 token 數。"""
    logp = log_softmax(model(ids[:, :-1]), dim=-1)[0]
    nll = -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]][start - 1 :]
    return nll.sum().item(), nll.numel()


torch.manual_seed(0)
for name, temperature, repeats in [("greedy", 0.0, 1), ("temperature 1.0", 1.0, 3)]:
    total, count = 0.0, 0
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        for _ in range(repeats):
            out = generate_cached(model, ids, 150, temperature=temperature)
            s, n = perplexity_after(out, ids.shape[1])
            total, count = total + s, count + n
    print(f"{name}：模型自己寫的 {count} 個 token，perplexity {math.exp(total / count):.2f}")
```

M4 Max 上的輸出（兩次執行相同）：

```text
greedy：模型自己寫的 600 個 token，perplexity 1.65
temperature 1.0：模型自己寫的 1800 個 token，perplexity 5.75
```

greedy 寫出來的字 perplexity 只有 1.65，比真正的故事（驗證集上約 5.4）低得多，但第 6 章第 1 集說過，greedy 容易卡進重複。每一步都挑模型最有把握的 token，對模型自己當然最不意外。抽樣寫出來的是 5.75，和驗證集的數字接近：模型照自己的機率寫，平均意外的程度就是它對語言的不確定程度。

單看 perplexity，會得到「重複的廢話比真實的故事好」。它也看不出答案對不對：模型可以把「巴黎是德國的首都」說得很流暢。

### 那該用什麼

perplexity 並沒有失去用處：同一個 tokenizer、同一份資料、只改訓練設定時，它是最便宜也最靈敏的指標，前兩部都是這樣用的。但要回答「模型會不會做某件事」，就得直接讓它做那件事，並用事先定好的標準打分。下一集開始這麼做。

## 重點整理

- perplexity 是每個 token 的數字，tokenizer 切得越細數字越低；不同 tokenizer 的模型要比 bits per byte：總 \\( -\log p \\) 除以 byte 數再除以 \\( \ln 2 \\)。
- 標準模型在驗證集上約 0.59 bits/byte；byte unigram 的每 byte perplexity（21.9）看起來比 token unigram 的每 token perplexity（332.6）小，但 bits/byte 是 4.45 對 2.03，結果相反。
- 同一個模型在不同領域的文字上，perplexity 可以從 5.7 到 13068.6：它反映文字離訓練資料多遠，所以比較模型要用同一份資料。
- greedy 寫出的文字 perplexity 只有 1.65（抽樣是 5.75），低 perplexity 不代表文字好，也看不出事實對不對。
- perplexity 適合在同 tokenizer、同資料下比較訓練設定；想知道模型會不會做某件事，要用 benchmark。
