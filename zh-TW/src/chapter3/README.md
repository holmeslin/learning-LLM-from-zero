# Tokenizer：從 bytes 到 token

模型只認得數字，不認得文字。這一章把文字變成一串整數（token id），再把整數變回文字：這個東西叫 **tokenizer**。我們用的方法是 BPE（byte-pair encoding）：從 256 個 byte 出發，反覆把最常一起出現的兩個相鄰 token 合併成一個新 token，直到詞彙表夠大。

全章自己動手寫：pre-tokenization、統計、合併、encode、decode、處理大檔，最後把 TinyStories 轉成本書的標準語料 token 檔，第 5 章訓練用的就是它。中間還會拿一段古文示範中文的 BPE 長什麼樣。

這一章對應 CS336 Assignment 1 的 tokenizer 部分與 Lecture 1 後半，見 <https://cs336.stanford.edu>。
