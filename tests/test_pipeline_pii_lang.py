"""第 13 章第 3～5 集：語言辨識、個資遮蔽與（共用的）字元 n-gram 分類器。"""

import math
from collections import Counter

import pytest

from llmz.pipeline import NGramClassifier, char_ngrams, load_langid_samples, mask_pii, train_langid

# 範例都是保留給文件用的假資料：example.com、RFC 5737 的 IP、555 開頭的號碼


def test_mask_pii_emails():
    text = "Write to Mia.Lee+news@example.com, or bob_2@mail.example.co.uk."
    masked, counts = mask_pii(text)
    assert masked == "Write to <EMAIL>, or <EMAIL>."  # 句尾的句號不屬於 email
    assert counts == {"email": 2, "phone": 0, "ip": 0}
    for text in ["@username on social media", "no at sign.example.com", "user@localhost", "a @ b.com"]:
        assert mask_pii(text) == (text, {"email": 0, "phone": 0, "ip": 0})  # 反例：原樣回傳


def test_mask_pii_ipv4():
    masked, counts = mask_pii("Server 192.0.2.1 and 198.51.100.77, not 256.1.1.1 or 1.2.3 or 10.0.0.1.5.")
    assert masked == "Server <IP> and <IP>, not 256.1.1.1 or 1.2.3 or 10.0.0.1.5."  # 超過 255、少一段、多一段都不是
    assert counts["ip"] == 2
    assert mask_pii("see 203.0.113.9.")[0] == "see <IP>."  # 句尾的句號
    assert mask_pii("v2.10.0.3b")[1]["ip"] == 0  # 緊貼著英文字母的版本號


@pytest.mark.parametrize(
    "number",
    ["+1 (555) 010-1234", "555-010-1234", "555.010.1234", "(02) 2345-6789", "02-2345-6789", "0912-000-111",
     "+886 912 000 111", "+886912000111", "03 1234 5678", "01 23 45 67 89"],
)  # fmt: skip
def test_mask_pii_phone_positive(number):
    masked, counts = mask_pii(f"Call {number} today")
    assert masked == "Call <PHONE> today"
    assert counts["phone"] == 1


@pytest.mark.parametrize(
    "text",
    ["2026-11-03", "2026-11-03 12:30:45", "version 1.2.3","Price: 1,234,567 dollars", "ISBN-less 12345", "page 12 of 345",
     "3.14159265358979", "in 1999, 2000 and 2001", "0912000111", "12 345 6", "10 11 12 13 14 15 16 17", "100 200 300 400 500 600"],
)  # fmt: skip
def test_mask_pii_phone_negative(text):
    assert mask_pii(text) == (text, {"email": 0, "phone": 0, "ip": 0})


def test_mask_pii_mixed_and_idempotent():
    text = "Contact: a@b.org / 555-010-1234 / 192.0.2.1\n中文也一樣：請撥 0912-000-111。"
    masked, counts = mask_pii(text)
    assert masked == "Contact: <EMAIL> / <PHONE> / <IP>\n中文也一樣：請撥 <PHONE>。"
    assert counts == {"email": 1, "phone": 2, "ip": 1}
    assert mask_pii(masked) == (masked, {"email": 0, "phone": 0, "ip": 0})  # 遮蔽過的文字再遮蔽一次不會變


def test_char_ngrams_pads_words_and_drops_punctuation_and_digits():
    # 「ab」補成 " ab "，1 到 2 個字元的片段：a、b 各 1 個（單獨的空格不算），" a"、"ab"、"b " 各 1 個
    assert char_ngrams("ab", 2) == Counter({"a": 1, "b": 1, " a": 1, "ab": 1, "b ": 1})
    assert char_ngrams("AB, 12 !", 2) == char_ngrams("ab", 2)  # 大小寫不分；標點、數字、空白都只是分隔
    assert char_ngrams("a b", 3) == Counter({"a": 1, "b": 1, " a": 1, "a ": 1, " b": 1, "b ": 1, " a ": 1, " b ": 1})  # 每個詞各自補空格
    assert char_ngrams("中文", 2)["中文"] == 1  # 中文沒有空格，一段連續的字就是一個「詞」
    assert char_ngrams("...", 3) == Counter()


def test_ngram_classifier_log_likelihood_matches_hand_computation():
    # n=1、alpha=1：x 看過 a 兩次、b 一次，y 看過 a 一次、b 兩次；詞彙表 {a, b} 共 2 個，每個標籤總共 3 個
    model = NGramClassifier.train({"x": ["aab"], "y": ["abb"]}, n=1, alpha=1)
    ll = model.log_likelihoods("aab")  # 片段：a 兩個、b 一個
    assert ll["x"] == pytest.approx(2 * math.log(3 / 5) + math.log(2 / 5))  # x：P(a) = (2+1)/(3+2)，P(b) = (1+1)/(3+2)
    assert ll["y"] == pytest.approx(2 * math.log(2 / 5) + math.log(3 / 5))
    assert model.predict("a")[0] == "x"
    assert model.predict("b")[0] == "y"
    label, posterior = model.predict("b")
    assert posterior == pytest.approx(1 / (1 + math.exp(model.log_likelihoods("b")["x"] - model.log_likelihoods("b")["y"])))


def test_ngram_classifier_ignores_unknown_ngrams_and_says_unknown_when_nothing_is_known():
    model = NGramClassifier.train({"en": ["the cat sat"], "de": ["die katze sass"]}, n=2)
    assert model.log_likelihoods("the cat qqq") == model.log_likelihoods("the cat")  # 沒看過的片段不影響比較
    assert model.predict("") == ("unknown", 0.0)
    assert model.predict("12345 !!!") == ("unknown", 0.0)
    assert model.predict("日本語") == ("unknown", 0.0)  # 訓練資料裡沒有任何一個片段出現過


def test_predict_lines_votes_by_letters_per_line():
    model = NGramClassifier.train({"x": ["aaaa aaaa"], "y": ["bbbb bbbb"]}, n=1)
    assert model.predict_lines("aaaa\nbb\nbb\nbb") == "y"  # 每行一票，票數是字母數：y 有 6 票、x 有 4 票
    assert model.predict_lines("aaaa\n1234\n!!\n日本語") == "x"  # 沒有任何看過的片段的行（unknown）不投票
    assert model.predict_lines("") == "unknown"
    test = load_langid_samples("test")
    page = "Home\nNews\nLogin\n" + "\n".join(test["zh"][:3])  # 英文的導覽列夾在中文頁面的最前面
    assert train_langid().predict_lines(page) == "zh"


def test_ngram_classifier_save_and_load(tmp_path):
    model = NGramClassifier.train({"en": ["the cat sat on the mat"], "de": ["die katze sass auf der matte"]}, n=3, alpha=0.5)
    model.save(tmp_path / "m.json")
    loaded = NGramClassifier.load(tmp_path / "m.json")
    assert (loaded.n, loaded.alpha) == (3, 0.5)
    for text in ["the mat", "die matte", "zzz"]:
        assert loaded.log_likelihoods(text) == model.log_likelihoods(text)
        assert loaded.predict(text) == model.predict(text)


def test_langid_samples_are_split_and_balanced():
    train, test = load_langid_samples("train"), load_langid_samples("test")
    assert set(train) == set(test) == {"en", "zh", "ja", "ko", "ru", "de", "fr", "es", "pt", "it"}
    assert all(len(texts) >= 10 for texts in train.values())
    assert all(len(texts) == 6 for texts in test.values())
    assert not {t for ts in train.values() for t in ts} & {t for ts in test.values() for t in ts}  # 訓練與測試沒有重複


def test_langid_accuracy_on_original_held_out_sentences():
    model = train_langid()
    test = load_langid_samples("test")
    wrong = [(lang, text) for lang, texts in test.items() for text in texts if model.predict(text)[0] != lang]
    assert not wrong, wrong
    assert model.predict("Bonjour, comment allez-vous aujourd'hui ?")[0] == "fr"  # 不在任何一個檔案裡的句子
    assert model.predict("簡單來說，這個函式會回傳一個字典。")[0] == "zh"
