# 資料混合與合成資料

## 本集目標

知道真的訓練資料不是「一個檔案」，而是幾個來源按比例混合；比例怎麼訂、一份資料重複看幾遍沒關係，以及合成資料能補什麼、會出什麼問題。這一集只用敘述加幾個小算式，不加新的函式。

## 正文

### 一個來源不夠

上一集最後有兩個數字：只看過兒童故事的標準模型，在網頁上的 loss 是 7.39；新模型只在 10 萬個 token 的網頁上練 50 步，驗證 loss 就到 5.54。模型學到什麼，是由它看過的資料決定的。想要一個模型既會寫故事、又會回答網頁上的問題、也會寫程式，訓練資料就要同時有這幾種東西。

做法是每個來源各自走一條管線（網頁走本章的管線，書、程式碼、維基百科走各自適合的那一條），然後在訓練時按比例混合：每個 batch 裡，有多少筆來自哪個來源，由「權重」決定。

### 權重不等於資料量

GPT-3 的論文（Brown 等人，2020，arXiv 2005.14165）第 2.2 節的表列出五個來源，每個有資料量、在訓練混合裡的權重，以及訓練 3000 億個 token 時每個來源被看了幾遍。論文寫明資料集不是照大小比例取樣的。把數字照抄進來，算每個來源占資料量的比例，與它的權重差多少：

```python
sources = {  # (資料量，單位十億個 token；訓練混合的權重；看了幾遍)，出處：GPT-3 論文的表 2.2
    "Common Crawl（過濾後）": (410, 0.60, 0.44),
    "WebText2": (19, 0.22, 2.9),
    "Books1": (12, 0.08, 1.9),
    "Books2": (55, 0.08, 0.43),
    "Wikipedia": (3, 0.03, 3.4),
}
total = sum(size for size, _, _ in sources.values())
for name, (size, weight, epochs) in sources.items():
    share = size / total
    print(f"{name}：占資料量 {share:.1%}，占訓練混合 {weight:.0%}（是資料量比例的 {weight / share:.2f} 倍），看了 {epochs} 遍")
```

執行結果：

```text
Common Crawl（過濾後）：占資料量 82.2%，占訓練混合 60%（是資料量比例的 0.73 倍），看了 0.44 遍
WebText2：占資料量 3.8%，占訓練混合 22%（是資料量比例的 5.78 倍），看了 2.9 遍
Books1：占資料量 2.4%，占訓練混合 8%（是資料量比例的 3.33 倍），看了 1.9 遍
Books2：占資料量 11.0%，占訓練混合 8%（是資料量比例的 0.73 倍），看了 0.43 遍
Wikipedia：占資料量 0.6%，占訓練混合 3%（是資料量比例的 4.99 倍），看了 3.4 遍
```

Common Crawl 占資料量的 8 成多，訓練混合只給它 60%，而且 3000 億個 token 的訓練裡只看了 0.44 遍，一遍都沒看完；Wikipedia 只有 0.6% 的資料量，權重是 3%，被看了 3.4 遍。權重是人訂的，不是資料量的結果：大的來源被打折，小的來源被放大。本章的篩選會讓來源變小，剩下的資料能不能給高權重，要看下一節的重複次數。

### 重複看幾遍可以

權重大於資料量比例，就表示那個來源會被重複看。網頁資料很多，重複不成問題；好的來源通常小，重複就是日常。重複幾遍還有用？Muennighoff 等人（2023，arXiv 2305.16264）在資料不夠的設定下比較過：計算量固定時，把資料重複用到 4 遍，loss 與用不重複的新資料相比幾乎沒差；再多，多花的計算量帶來的好處會逐漸掉到零。

第 10 集輸出的 `web.bin` 只有 108,088 個 token。第 5 章標準訓練一共看 \\( 5000 \times 32 \times 256 \\) 個 token，如果要網頁占其中一部分，要重複看多少遍：

```python
run_tokens = 5000 * 32 * 256  # 第 5 章的標準訓練：步數 × batch × context
web_tokens = 108_088  # 第 10 集的 web.bin
for weight in [0.01, 0.05, 0.10]:
    needed = weight * run_tokens
    print(f"網頁占 {weight:.0%}：要 {needed:,.0f} 個 token，這份檔案要重複看 {needed / web_tokens:.1f} 遍")
print(f"最多看 4 遍的話，網頁只能占 {4 * web_tokens / run_tokens:.2%}")
```

執行結果：

```text
網頁占 1%：要 409,600 個 token，這份檔案要重複看 3.8 遍
網頁占 5%：要 2,048,000 個 token，這份檔案要重複看 18.9 遍
網頁占 10%：要 4,096,000 個 token，這份檔案要重複看 37.9 遍
最多看 4 遍的話，網頁只能占 1.06%
```

想讓網頁占 10%，這個檔案要看 38 遍，遠超過 4 遍。所以混合比例不只是「想要多少」，也受每個來源有多少資料限制：資料少的來源，權重能開的上限很低，要提高，只能找更多資料（再下載更多 WARC 檔）。

### 比例怎麼訂

沒有公式。常見的做法是用小模型試：每一組比例各練一個小模型，在第 12 章的評估上比較，挑最好的，再放大。第 10 章量過，小模型上的趨勢可以外推，但外推有誤差，混合比例也不例外。

DoReMi（Xie 等人，2023，arXiv 2305.10429）把這件事自動化：先用一種叫 group DRO 的方法，訓練一個小的代理模型（proxy），讓它在各個領域之間找出一組權重，過程不需要知道下游的任務；再照這組權重重新取樣資料，訓練大的模型。論文用 2.8 億個參數的代理模型，替 80 億個參數（30 倍大）的模型選權重；和用 The Pile 預設的領域權重訓練的模型相比，平均 few-shot 準確率高 6.5 個百分點，達到基準準確率所需的訓練步數少了 2.6 倍。

不管用哪種方法，「最好」都是對照你選的評估來說的：評估沒涵蓋的能力，混合比例裡就沒有它的位置（第 12 章）。

### 合成資料

合成資料是讓另一個模型寫出來的訓練資料。你已經用了好幾章：標準語料 TinyStories（Eldan 與 Li，2023，arXiv 2305.07759）是 GPT-3.5 與 GPT-4 生成的短篇故事，只用三到四歲小孩聽得懂的字。論文說，這份資料可以用來訓練與評估小得多的模型（不到 1000 萬個參數，或只有一個 transformer block），它們仍然寫得出好幾段、通順連貫的故事。本書下載的是它的 GPT-4 版本（檔名 `TinyStoriesV2-GPT4-train.txt`）。

合成資料有什麼不同？拿第 10 集的 `web.bin` 對照標準語料開頭同樣長的一段，數用了幾種不同的 token，與最常見的 100 種占多少：

```python,norun
import numpy as np

from llmz.data import load_tokens

web = np.asarray(load_tokens("data/cc/web.bin"))
stories = np.asarray(load_tokens("data/train.bin")[: len(web)])  # 標準語料開頭同樣長的一段

for name, tokens in [("標準語料（合成的故事）", stories), ("網頁（第 10 集的輸出）", web)]:
    counts = np.sort(np.bincount(tokens, minlength=10000))[::-1]  # 每種 token 出現幾次，由多到少
    print(f"{name}：{len(tokens):,} 個 token 用了 {int((counts > 0).sum()):,} 種，最常見的 100 種占 {counts[:100].sum() / len(tokens):.1%}")
```

M4 Max 上的輸出（每次相同）：

```text
標準語料（合成的故事）：108,088 個 token 用了 3,587 種，最常見的 100 種占 64.6%
網頁（第 10 集的輸出）：108,088 個 token 用了 5,080 種，最常見的 100 種占 45.1%
```

故事用的 token 種類比網頁少三成，最常見的 100 種就占了 6 成多。這是設計出來的：窄、乾淨、好學。不過 tokenizer 本來就是在這些故事上訓練的，網頁被拆成更細碎的片段，所以兩個數字的差距有一部分來自 tokenizer，不全是資料。

合成資料的用處，大致是三件事：

- **控制**：要什麼主題、什麼難度、什麼格式，在 prompt 裡寫。TinyStories 用「三到四歲」控制詞彙。
- **補缺**：網頁上少的東西（某種題型、某種語言、某種格式），可以請模型寫出來。phi-1（Gunasekar 等人，2023，arXiv 2306.11644）是一個例子：用 GPT-3.5 生成 10 億個 token 的「教科書」與習題，加上 60 億個 token 從網頁篩出來的「教科書品質」資料，訓練一個 13 億個參數的模型，HumanEval（一個寫程式的評測）pass@1 是 50.6%。從網頁篩出高品質的資料，就是第 7 集品質分類器的同一個想法；合成資料補的是網頁裡不夠整齊的部分。
- **格式**：網頁上少見的成對格式（問題與答案、指示與回應），可以請模型寫出來；第 14 章會談到。

也有問題：

- **分布是生成模型的，不是世界的**。Shumailov 等人（2023，arXiv 2305.17493）發現，用模型生成的內容訓練下一個模型，會讓結果出現不可逆的缺陷：原本內容分布的尾巴消失，他們叫它 model collapse。合成資料混進真實資料的比例，是要管的。
- **還是要去重、篩品質**。合成資料也會重複：第 12 章量過，標準語料的驗證集就有 8.2% 的 13-gram 出現在訓練集。本章的管線一樣用得上。
- **評估要小心**。如果生成資料的模型看過評測題目，那些題目的內容也可能跑進你的訓練資料；第 12 章第 5 集的 n-gram 重疊偵測，對合成資料也要做。

### 把本章接起來

網頁走過本章的管線：WARC 解出頁面，HTML 轉文字，去重與近似去重，辨識語言，Gopher 規則，品質分類器，遮蔽 PII，輸出成 token 檔。每個來源都走自己的管線，變成各自的 token 檔，再由取樣器依權重決定每個 batch 的組成，交給第 5 章的訓練程式。本章寫出了前半段；後半段的取樣器，是這一集的練習。

## 重點整理

- 真的訓練資料是多個來源按權重混合：每個來源各走一條管線，訓練時才混。GPT-3 的混合裡，Common Crawl 占資料量 82% 卻只有 60% 的權重、看了 0.44 遍；Wikipedia 占 0.6% 卻有 3%、看了 3.4 遍。
- 權重大於資料量比例就會重複看。在資料不夠的設定下，重複到 4 遍和不重複幾乎一樣好（Muennighoff 等人）；`web.bin` 想占標準訓練的 10%，要重複 38 遍，所以資料少的來源權重上限很低。
- 比例靠小模型實驗、或像 DoReMi 用代理模型自動調；「最好」是對你選的評估而言。
- 合成資料可以控制主題與難度、補缺；問題是分布是生成模型的（model collapse）、還是要去重篩品質、評估要防污染。
- 本章的管線：WARC → 文字 → 去重 → 語言 → Gopher → 品質 → PII → token 檔。

## 自我挑戰

這一章放進 `src/llmz/` 的是 `src/llmz/pipeline.py` 的 `iter_warc_records`、`split_http_response`、`decode_html`、`html_to_text`、`iter_pages`、`write_pages`、`read_pages`、`char_ngrams`、`NGramClassifier`（`train`、`log_likelihoods`、`predict`、`predict_lines`、`save`、`load`）、`load_langid_samples`、`train_langid`、`mask_pii`、`gopher_check`、`hash_ngrams`、`QualityClassifier`（`forward`、`score`）、`pad_docs`、`binary_cross_entropy`、`train_quality`、`line_hash`、`count_lines`、`dedup_lines`、`shingles`、`jaccard`、`minhash_signature`、`lsh_probability`、`lsh_candidates`、`verify_pairs`、`drop_near_duplicates`、`run_pipeline`、`keep_language`、`keep_ascii`、`keep_gopher`、`mask_pii_pages`、`pareto_keep`、`keep_by_quality`、`write_corpus_text`。把它們的本體換成 `raise NotImplementedError`（`Page`、`StepReport`、常數與底線開頭的輔助函式留著），執行：

```bash
uv run pytest tests/test_pipeline.py tests/test_pipeline_pii_lang.py
```

兩個測試檔都要通過：`tests/test_pipeline.py` 涵蓋 WARC、HTML、Gopher 規則、品質分類器、去重、MinHash、LSH 與本集的管線；`tests/test_pipeline_pii_lang.py` 涵蓋 PII 遮蔽、字元 n-gram 與語言辨識。自己寫回來讓測試通過。

再試一題：第 10 集用 `keep_ascii` 補語言辨識沒有「都不是」的洞。替 `keep_language` 加一個參數 `min_seen`：頁面的字元片段（`char_ngrams`）有多少比例在 `model.vocab` 裡，低於 `min_seen` 就不算這個語言。在 817 頁上量：`min_seen` 設多少，可以擋下第 10 集那一頁捷克文，又不會誤殺通過 Gopher 規則的英文頁面？
