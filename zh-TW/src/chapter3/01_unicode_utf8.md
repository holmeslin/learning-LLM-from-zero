# 字串、Unicode 與 UTF-8

## 本集目標

知道一個字元在電腦裡是一個整數（Unicode 碼位），存檔時又變成 1 到 4 個 byte（UTF-8），並會用 `encode`／`decode` 在兩者之間轉換。

## 正文

### 字元是數字

Python 的 `ord` 把字元變成它的 Unicode 碼位，`chr` 變回來：

```python
print(ord("A"), chr(65))
print(ord("你"), ord("🐱"))
```

執行結果：

```text
65 A
20320 128049
```

Unicode 給全世界每個字元一個號碼，目前有十幾萬個。但模型的詞彙表不可能直接用「每個 Unicode 字元一個 token」：太多，而且大部分一輩子看不到幾次。

### 存檔時用的是 byte

檔案裡放的不是碼位，而是 **UTF-8** 編碼後的 **byte**（位元組，0 到 255 的整數）。`str.encode` 做這件事：

```python
s = "Hi 你好"
b = s.encode("utf-8")
print(b)
print(list(b))
print(len(s), len(b))
print(b.decode("utf-8"))
```

執行結果：

```text
b'Hi \xe4\xbd\xa0\xe5\xa5\xbd'
[72, 105, 32, 228, 189, 160, 229, 165, 189]
5 9
Hi 你好
```

`b'...'` 是 Python 的 bytes 型別，印出來時英文字母照顯示、其他的用 `\x` 加十六進位。`list(b)` 讓你看到它其實就是一串 0～255 的整數。5 個字元變成了 9 個 byte，因為每個中文字佔 3 個 byte。`.decode("utf-8")` 把 bytes 轉回字串。

### 一個字元佔幾個 byte

```python
for ch in "A é 你 🐱".split():
    print(ch, len(ch.encode("utf-8")), list(ch.encode("utf-8")))
```

執行結果：

```text
A 1 [65]
é 2 [195, 169]
你 3 [228, 189, 160]
🐱 4 [240, 159, 144, 177]
```

英文 1 個 byte，歐洲重音字母 2 個，中日韓 3 個，表情符號 4 個。UTF-8 的設計讓英文檔案和老式的 ASCII 完全相容，代價是其他語言比較佔空間，這會在第 13 集影響「一個 token 平均代表幾個字」。

### 切到一半會壞掉

一個多 byte 字元只拿前面幾個 byte，就不是合法的 UTF-8：

```python
print(bytes([228, 189]).decode("utf-8", errors="replace"))
```

執行結果：

```text
�
```

`errors="replace"` 讓它不要報錯，改成印出替代字元 `�`。本章的 tokenizer 以 byte 為最小單位，所以模型有可能吐出「半個字」，第 10 集的 decode 就靠這個參數撐住。

### 為什麼從 byte 出發

byte 只有 256 種，任何語言、任何符號、任何亂碼都能表示，不會有「詞彙表裡沒有這個字」的問題。本章的 BPE 就從這 256 個 byte 開始，把常見的組合黏成更長的 token。

## 重點整理

- 字元在 Unicode 裡是一個整數（`ord`／`chr`）；存檔時用 UTF-8 變成 1～4 個 byte（`encode`／`decode`）。
- 英文 1 byte、中文 3 byte、表情符號 4 byte。
- 不完整的 byte 序列 decode 會壞，`errors="replace"` 讓它變成 `�`。
- 256 種 byte 能表示一切，是 BPE 的起點。
