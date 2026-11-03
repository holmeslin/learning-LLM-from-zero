# 完整生成函式

## 本集目標

把前三集組成 `generate(model, tokenizer, prompt, ...)`：文字進、文字出，並用第 5 章訓練好的模型寫幾個故事。

## 正文

### 組裝

```python,ignore
{{#include ../../../src/llmz/generate.py:generate}}
```

逐行：

- `next(model.parameters()).device`：從模型的第一個參數看它在哪個 device，輸入就放同一個地方。
- `tokenizer.encode(prompt)` 把開頭文字變 id；`special_to_id` 查 `<|endoftext|>` 的 id，生成到它就停。
- 迴圈每一步只餵最後 `context_length` 個 token（模型的 RoPE 表只有這麼長），取最後一個位置的 logits，`temperature == 0` 用 greedy，否則 `sample_next`。
- 最後 `decode` 整串 id，包含開頭的 prompt。

整個函式 `@torch.no_grad()`。注意這個實作每一步都把整段序列重算一遍，生成 100 個 token 就做了 100 次 forward，前面的位置算了又算；第 11 章的 KV cache 會把重複的部分存起來。現在先求對。

### 載入第 5 章的模型

```python,norun
import torch

from llmz.device import get_device
from llmz.generate import generate
from llmz.model import ModelConfig, TransformerLM
from llmz.tokenizer import Tokenizer

device = get_device()
model = TransformerLM(ModelConfig()).to(device)
model.load_state_dict(torch.load("runs/base/model.pt", map_location=device))
model.eval()
tokenizer = Tokenizer.load("data/tokenizer.json")

torch.manual_seed(0)
for prompt in ["Once upon a time, there was a little girl named Lily.", "The dog saw a big"]:
    print(generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_p=0.9))
    print("---")
```

`ModelConfig()` 的預設就是訓練時的設定，所以直接建；`map_location=device` 讓存檔時在 MPS 上的參數直接載到目前的 device。這段需要 `runs/base/model.pt`，所以不會被自動驗證；在你的機器上跑出來的故事會和下面不同（MPS 的隨機數不保證一致），但風格一樣。

M4 Max 上的輸出（某一次）：

```text
Once upon a time, there was a little girl named Lily. Lily loved to eat food, especially fruits. One day, she found a big, round squash in the garden. She was very happy and wanted to eat it.
Lily took the squash to her mom. Her mom said, "Lily, don't eat that squash. It's not good for you." But Lily did not listen. She ate the squash all day.
Later, Lily's friend, Tom, came over to play. Tom saw the squash and said, "Wow, you are so big, but you have to be careful with it." Lily was happy and ran to Tom
---
The dog saw a big pile of trash and stopped digging. The waste had dirt and dirt. The dog thought the waste was dead. He wanted to keep the waste clean.
The dog went to the waste bin and found a nice toy. He brought the toy back to the dog. The dog was happy. The dog felt good to help.
The dog's owner, a nice lady, saw the waste and said, "You should help someone in need. But you should not take things that are not yours." The dog listened to the lady and did not take the waste.
The moral of the story is
---
```

### 三種設定比較

同一個開頭，不同的取樣設定：

開頭都是 `One day, a little bird`：

| 設定 | 生成（M4 Max 某一次，節錄） | 效果 |
| --- | --- | --- |
| `temperature=0` | One day, a little bird named Tim wanted to soar high in the sky. He saw a big tree with a lot of fruit. Tim thought it would be fun to fly up and see the fruit. Tim flew to the fruit and said, "Hello, fruit! Can I have some fruit?" | 每次都一樣；通順，但「fruit」一直出現，再長一點就會開始重複句型 |
| `temperature=1.0, top_p=1.0` | One day, a little bird was watching the sunrise. The bird saw the sunrise and wanted to make it yummy. So, the bird flew down and picked up a big, redhew. "Stop, bird!" the little bird said, looking at the sunrise. The bird went around, still holding theZoom. | 變化最大，但 `redhew`、`theZoom` 這種從尾巴抽到的怪字會出現 |
| `temperature=0.8, top_p=0.9` | 上面兩個故事 | 本書預設：通順、有變化、幾乎沒有怪字 |

自己把參數改一改，用同一個種子比較。這個模型只有 2270 萬參數、看了 4100 萬個 token，能寫出有頭有尾的童話，已經很不錯；它不懂事實、不會推理，再多生幾個也會看到自相矛盾的句子。

## 重點整理

- `generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p)`：encode → 迴圈（餵最後 `context_length` 個、取最後位置 logits、挑 token、遇到結束 token 停）→ decode。
- 載入：`TransformerLM(ModelConfig())`、`load_state_dict(torch.load(..., map_location=device))`、`eval()`。
- 預設 temperature 0.8、top-p 0.9；每步重算整段序列，第 11 章用 KV cache 改善。
