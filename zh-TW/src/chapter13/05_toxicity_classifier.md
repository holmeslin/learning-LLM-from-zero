# 有害內容分類

## 本集目標

知道資料管線裡「有害內容分類器」站在哪裡、做什麼，以及它為什麼是管線裡最容易出事的一步：字表、外部服務、自己訓練的分類器三種做法各有什麼問題；分數與門檻怎麼決定留下或丟掉什麼；訓練資料怎麼決定它會誤殺誰。用第 3 集的 `NGramClassifier` 與一個完全沒有不當內容的玩具例子示範。

## 正文

### 它在管線的哪裡

網頁裡有仇恨言論、騷擾、成人內容。模型訓練時看過的東西，生成時就有可能出現，所以很多管線在訓練前加一道篩選：給每份文件（或每個句子）一個「有多有害」的分數，超過門檻的處理掉。位置通常在語言辨識與 PII 遮蔽之後、品質篩選之前：前面的步驟把文件整理乾淨，這一步決定哪些內容完全不要。

「處理掉」有不同的粒度：整頁丟掉，或只丟掉分數高的句子，其餘留下。分數怎麼來，有三種做法，都有人在大型資料集用過：

| 做法 | 例子 | 怎麼運作 |
| --- | --- | --- |
| 字表 | C4（Raffel 等人 2020） | 一頁裡只要出現「List of Dirty, Naughty, Obscene or Otherwise Bad Words」上的任何一個字，整頁丟掉 |
| 外部服務 | Gopher 的 MassiveWeb（Rae 等人 2021） | 沒有通過 Google SafeSearch 的頁面丟掉 |
| 自己訓練的分類器 | Dolma（Soldaini 等人 2024） | 用 Jigsaw Toxic Comments（有人工標註的論壇留言）訓練 fastText 分類器，分成「hate」與「NSFW」兩個，對每個句子打分，超過門檻的句子丟掉 |

Gopher 的論文特別寫到，他們選 SafeSearch 而不是手動的字表，因為字表已經被發現會不成比例地丟掉和少數族群有關、本身並不冒犯的內容。這個發現來自 Dodge 等人（2021）對 C4 的分析：C4 的字表過濾，不成比例地丟掉了少數族群用的英文（例如非裔美國人的英文），以及談論 LGBTQ+ 身分的文字。Welbl 等人（2021）在 DeepMind 的實驗也看到類似的事：用毒性分數過濾訓練資料或解碼結果，自動指標會變好，但模型對少數群體相關文字與方言的涵蓋也變差。

這一集的重點就在這裡：**分類器是一個決定「什麼聲音留在資料裡」的工具**，它的錯誤不是隨機的。

### 用發明的字做玩具

本書不放不當內容，所以用一個玩具：遊戲聊天室，裡面的「罵人話」是我們發明的三個字：`zorp`、`blorf`、`snurgle`。訓練與測試的句子全是原創。分類器直接用第 3 集的 `NGramClassifier`，標籤換成 `flagged` 與 `ok`，一行程式都不用改。先比較「字表」（發明的三個字有沒有出現）與分類器：

```python
import math

from llmz.pipeline import NGramClassifier, char_ngrams

FLAGGED = [
    "you are such a zorp, nobody wants you on this team",
    "what a blorf, you lost us the game again",
    "shut up you big snurgle and go away",
    "only a total zorp would play like that",
    "get out of here, you worthless blorf",
    "you snurgle, delete the game and never come back",
    "stupid zorp, you ruin every match",
    "blorf blorf blorf, nobody likes you",
    "go away you filthy snurgle, you are the worst",
    "i hope you lose, you rotten zorp",
    "what a pathetic blorf you are",
    "you absolute snurgle, quit now",
]
OK = [
    "good game everyone, that was a close match",
    "nice shot, you saved us in the last round",
    "can someone help me with the second level",
    "i will be back in five minutes, please wait for me",
    "thanks for the invite, this team is fun",
    "the new map looks great, i love the colors",
    "does anyone know how to unlock the blue sword",
    "sorry i was late, my internet was slow",
    "well played, see you all tomorrow",
    "who wants to trade a shield for a potion",
    "the boss is hard, but we can do it together",
    "congratulations on reaching level twenty",
]
OK_MONSTERS = [  # 第三部分才會用到
    "a zorp is a small green monster in the first level",
    "the blorf lives in the cave and guards a treasure chest",
    "i caught a snurgle and sold it for ten coins",
]
TEST = [  # (文字, 真正的標籤)
    ("you big zorp, nobody likes you here", "flagged"),
    ("what a rotten snurgle you are", "flagged"),
    ("go away blorf, we do not want you", "flagged"),
    ("you worthless zoorp, quit the game", "flagged"),  # 拼錯一個字母
    ("such a pathetic blorff, you ruin everything", "flagged"),
    ("stupid snurrgle, get out", "flagged"),
    ("you are the worst player, quit now", "flagged"),  # 沒有用到那三個字
    ("nobody wants you on this team, go away", "flagged"),
    ("a zorp is a green monster with three eyes in the second level", "ok"),  # 遊戲裡的怪物名稱
    ("the wiki says a blorf drops a blue sword when you defeat it", "ok"),
    ("i found a snurgle near the lake, it dropped a shield", "ok"),
    ("thanks for the help, that boss was hard", "ok"),
    ("good game, see you tomorrow on the new map", "ok"),
    ("can someone help me unlock the sword", "ok"),
    ("sorry i lost the match, my internet was slow", "ok"),
    ("nice shot, that was a close game", "ok"),
]
BLOCKLIST = {"zorp", "blorf", "snurgle"}


def blocklist_flags(text):
    return any(word in BLOCKLIST for word in text.replace(",", " ").split())


def score(model, text):
    """每個片段平均的 log 機率差：0 是兩邊一樣像，正的偏向 flagged，負的偏向 ok。"""
    ll = model.log_likelihoods(text)
    return (ll["flagged"] - ll["ok"]) / sum(char_ngrams(text, model.n).values())


def report(name, flags):
    caught = sum(f and label == "flagged" for f, (_, label) in zip(flags, TEST))
    missed = sum((not f) and label == "flagged" for f, (_, label) in zip(flags, TEST))
    wrong = sum(f and label == "ok" for f, (_, label) in zip(flags, TEST))
    print(f"{name}：抓到 {caught}/{caught + missed}，誤殺 {wrong}")


model = NGramClassifier.train({"flagged": FLAGGED, "ok": OK})
print("--- 字表與分類器 ---")
report("字表", [blocklist_flags(text) for text, _ in TEST])
report("分類器 predict", [model.predict(text)[0] == "flagged" for text, _ in TEST])

print("--- 不同的門檻 ---")
scores = [score(model, text) for text, _ in TEST]
for threshold in [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
    report(f"門檻 {threshold:4}", [s > threshold for s in scores])

print("--- 加了 3 句怪物名稱的 ok 例子之後 ---")
model = NGramClassifier.train({"flagged": FLAGGED, "ok": OK + OK_MONSTERS})
scores = [score(model, text) for text, _ in TEST]
for threshold in [-0.1, 0.0, 0.5]:
    report(f"門檻 {threshold:4}", [s > threshold for s in scores])
```

執行結果：

```text
--- 字表與分類器 ---
字表：抓到 3/8，誤殺 3
分類器 predict：抓到 8/8，誤殺 1
--- 不同的門檻 ---
門檻 -0.5：抓到 8/8，誤殺 4
門檻 -0.1：抓到 8/8，誤殺 2
門檻  0.0：抓到 8/8，誤殺 1
門檻  0.1：抓到 8/8，誤殺 0
門檻  0.5：抓到 8/8，誤殺 0
門檻  1.0：抓到 3/8，誤殺 0
--- 加了 3 句怪物名稱的 ok 例子之後 ---
門檻 -0.1：抓到 8/8，誤殺 1
門檻  0.0：抓到 8/8，誤殺 0
門檻  0.5：抓到 8/8，誤殺 0
```

### 字表與分類器

第一部分比較「字表」（發明的三個字有沒有出現）與分類器。字表只抓到 3 句：它不認得拼錯的 `zoorp`、`blorff`、`snurrgle`，也抓不到沒用那三個字的 `worst player, quit now`；而且把 3 句談怪物名稱的正常句子都當成罵人。分類器看的是字元片段，拼錯一個字母還是有 ` zo`、`orp` 這些片段，所以 8 句都抓到；`worst`、`quit now`、`nobody wants you` 這些片段在 `flagged` 的訓練句裡多、`ok` 裡沒有，也被抓到。它還是誤殺了 1 句。

### 分數與門檻

`predict` 回傳的機率不能拿來設門檻：第 3 集看到，naive Bayes 的機率幾乎都是 0 或 1。腳本裡的 `score` 換成「每個片段平均的 log 機率差」：除以片段數，句子的長短不會把它放大；有了連續的分數，門檻就是一個旋鈕。

第二部分把同一個測試集放在不同的門檻下。門檻低（`-0.5`），什麼都抓，也誤殺 4 句；門檻高（`1.0`），一句誤殺都沒有，但只抓到 3 句。中間（`0.1` 到 `0.5`）剛好把兩邊分開，但這是 16 句的玩具測試集；真的資料沒有這麼乾淨，兩邊的分數會重疊，每個門檻都在「多丟一些」與「多留一些」之間挑。這個取捨沒有標準答案，要看你比較怕哪一種：訓練資料裡留下有害的內容，或是把正常的內容丟掉。

### 資料決定一切

為什麼 `the wiki says a blorf drops a blue sword` 會被誤殺？因為訓練資料裡，`blorf` 只出現在 `flagged` 那一邊。分類器學到的不是「這是罵人」，而是「這個片段和標成 `flagged` 的句子一起出現」。第三部分把 3 句怪物名稱的 `ok` 例子加進訓練資料：門檻 `0.0` 的誤殺從 1 句降到 0 句，而且抓到的還是 8 句。

這不是玩具才有的現象。標註的人決定什麼算「有害」，訓練資料決定分類器把哪些片段和「有害」連在一起；如果某個身分、方言、話題在標成有害的資料裡出現得比較多（例如這些詞常常在被罵的留言裡出現），分類器就會把它們當成有害的訊號，不管那句話本身有沒有惡意。Dodge 與 Welbl 看到的就是這個：被誤殺的是少數群體自己寫的、或談論他們的文字。

### 本書的管線不做這一步

第 10 集組管線時，我們沒有這一步：沒有標註資料，也不想在書裡放不當內容；網頁裡的 PII 與語言問題我們量得出對錯，有害內容的對錯要靠人工標註，量不出來的數字不寫。真的要做的時候，介面就是這一集的：一個把文字變成分數的分類器（`NGramClassifier` 或別的），加一個門檻，決定整頁丟、丟句子，或只是在資料上標記，以後再決定。有幾件事要先想清楚：

- 標註資料誰標的、標準是什麼：這決定誤殺誰。
- 門檻訂在哪裡，並且量過它丟掉什麼：不只看丟了多少份文件，也要看丟掉的文件是哪一類。
- 只丟句子還是整頁：句子層級保留更多文字，但句子被丟掉之後，前後文就接不起來了。

## 重點整理

- 有害內容分類器給每份文件或每個句子一個分數，超過門檻就處理掉；位置在語言辨識與 PII 遮蔽之後、品質篩選之前。
- 三種做法：字表（C4：任何一個字出現就整頁丟）、外部服務（Gopher：Google SafeSearch）、自己訓練的分類器（Dolma：用 Jigsaw Toxic Comments 訓練 fastText，對句子打分）。
- 字表認不得拼錯的字，也分不出「提到」與「使用」；分類器看字元片段，拼錯還抓得到，但會學到訓練資料裡的偏差。
- 要設門檻，要用沒有被放大的連續分數（每個片段平均的 log 機率差），不是 naive Bayes 的後驗機率；門檻是「多丟一些」與「多留一些」的取捨。
- 分類器的錯誤不是隨機的：Dodge 等人與 Welbl 等人都發現，過濾會不成比例地丟掉少數群體寫的或談論他們的文字；要看資料是誰標的，並量過丟掉的是哪些。
