"""tools/checkcode.py 的測試：用一份小 markdown 走過每種標註。"""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("checkcode", Path(__file__).parent.parent / "tools" / "checkcode.py")
checkcode = importlib.util.module_from_spec(spec)
sys.modules["checkcode"] = checkcode  # dataclass 需要在 sys.modules 裡找到模組
spec.loader.exec_module(checkcode)

FIXTURE = '''# 測試

```python
print("hi")
```

執行結果：

```text
hi
```

```python
print("wrong")
```

執行結果：

```text
right
```

```python,ignore
not python at all
```

```python,error
raise ValueError("故意的")
```

```python,error
print("沒出錯")
```

```python,norun
def f(:
```

```python,norun
while True: pass
```

```python
import random
print(random.random())
```

執行結果（某一次）：

```text
0.5
```

```python,stdin=小明
print("你好，" + input())
```

執行結果：

```text
你好，小明
```

````markdown
```python
print("inside a nested fence, must not be collected")
```
````

```python,ignore
{{#include ../lib.py:exists}}
```

```python,ignore
{{#include ../lib.py:missing}}
```

```python,ignore
{{#include ../nothere.py}}
```
'''

LIB = """
# ANCHOR: exists
x = 1
# ANCHOR_END: exists
"""


def test_collect_and_check(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text(FIXTURE, encoding="utf-8")
    (tmp_path / "lib.py").write_text(LIB, encoding="utf-8")

    blocks, errors = checkcode.collect(src)
    assert [b.line for b in blocks] == [3, 13, 23, 27, 31, 35, 39, 43, 54, 70, 74, 78]
    assert sorted(e.split(": ", 1)[1] for e in errors) == ["../lib.py 裡沒有 anchor missing", "include 找不到檔案 ../nothere.py"]

    todo = [b for b in blocks if "ignore" not in b.attrs]
    results = {b.line: checkcode.check(b) for b in todo}
    assert results[3] is None
    assert results[13] is not None and "輸出與書上不一致" in results[13]
    assert results[27] is None
    assert results[31] == "預期會出錯，卻正常結束"
    assert results[35] is not None and "語法錯誤" in results[35]
    assert results[39] is None  # norun 只檢查語法，不會真的無窮迴圈
    assert results[43] is None and blocks[7].expected is None  # 「某一次」不比對
    assert results[54] is None


def test_main_exit_code(tmp_path: Path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.md").write_text('```python\nprint(1)\n```\n\n執行結果：\n\n```text\n1\n```\n', encoding="utf-8")
    assert checkcode.main(["checkcode", str(src)]) == 0
    (src / "bad.md").write_text('```python\nprint(1)\n```\n\n執行結果：\n\n```text\n2\n```\n', encoding="utf-8")
    assert checkcode.main(["checkcode", str(src)]) == 1
    assert "FAIL" in capsys.readouterr().out
