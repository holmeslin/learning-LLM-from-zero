"""後訓練（第 14 章）：chat template 與 loss mask、帶 mask 的 SFT loss、程式產生的指令資料、SFT 訓練迴圈。"""

import json
import os
import random
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor, nn

from llmz.evaluation import exact_match, sequence_logprob
from llmz.device import get_device
from llmz.inference import generate_cached
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW, clip_grad_norm_, lr_cosine_schedule
from llmz.tokenizer import Tokenizer

END = "<|endoftext|>"  # 標準語料的文件分隔符；在這裡當成「助理這一輪說完了」


# ANCHOR: chat_template
def chat_pieces(messages: list[dict], generation_prompt: bool = False) -> list[tuple[str, bool]]:
    """訊息列表 -> [(文字, 要不要算 loss)]。只有助理的回覆（加上結尾 <|endoftext|>）算 loss，提示的部分都不算。

    generation_prompt=True 時，最後多接一個 "Assistant:"，請模型從這裡開始寫回覆；平常訓練不用。
    """
    pieces = []
    for message in messages:
        if message["role"] == "user":
            pieces.append((f"User: {message['content']}\n", False))
        elif message["role"] == "assistant":
            pieces += [("Assistant:", False), (f" {message['content']}{END}", True)]
        else:
            raise ValueError(f"不認得的 role：{message['role']}")
    if generation_prompt:
        pieces.append(("Assistant:", False))
    return pieces


def encode_chat(tokenizer, messages: list[dict], generation_prompt: bool = False) -> tuple[list[int], list[bool]]:
    """訊息列表 -> (token id, loss mask)，兩個一樣長。每一段分開 encode 再接起來，mask 就是每段的 True 或 False 重複 token 數次。"""
    ids, mask = [], []
    for text, train in chat_pieces(messages, generation_prompt):
        piece = tokenizer.encode(text)
        ids += piece
        mask += [train] * len(piece)
    return ids, mask
# ANCHOR_END: chat_template


# ANCHOR: sft_loss
def pad_batch(examples: list[tuple[list[int], list[bool]]]) -> tuple[Tensor, Tensor]:
    """[(ids, mask), ...] -> ids (B, T)、mask (B, T)：短的往右補 0，補的位置 mask 是 False。
    模型是 causal 的，右邊補的 token 不會影響左邊，所以不用另外遮。"""
    longest = max(len(ids) for ids, _ in examples)
    ids = torch.zeros(len(examples), longest, dtype=torch.long)
    mask = torch.zeros(len(examples), longest, dtype=torch.bool)
    for row, (example_ids, example_mask) in enumerate(examples):
        ids[row, : len(example_ids)] = torch.tensor(example_ids)
        mask[row, : len(example_mask)] = torch.tensor(example_mask)
    return ids, mask


def sft_loss(model: nn.Module, ids: Tensor, mask: Tensor) -> Tensor:
    """mask 為 True 的 token 的平均 -log 機率（整個 batch 的 token 一起平均）。第 12 章的 sequence_logprob 加個負號、除以 token 數。
    shortcut: mask 全是 False 時沒有 token 可算，分母用 1 擋掉除以 0，loss 是 0；真的遇到代表資料有問題，要自己檢查。"""
    n_tokens = mask[:, 1:].sum().clamp(min=1)  # 第 0 個 token 沒有前文，永遠不算（和 sequence_logprob 一致）
    return -sequence_logprob(model, ids, mask).sum() / n_tokens
# ANCHOR_END: sft_loss


# ANCHOR: task_makers
class Task(NamedTuple):
    kind: str  # "add"、"subtract"、"count"、"color"、"who"
    question: str  # 使用者說的話
    answer: str  # 標準答案，一個字


NAMES = {"Tom": "He", "Ben": "He", "Sam": "He", "Max": "He", "Tim": "He", "Anna": "She", "Lily": "She", "Mia": "She", "Sue": "She", "Amy": "She"}
THINGS = ["apple", "ball", "cookie", "toy", "flower", "book", "shell", "hat", "kite", "cup"]
COLORS = ["red", "blue", "green", "yellow", "pink", "purple", "orange", "black", "white", "brown"]
NUMBERS = "zero one two three four five six seven eight nine ten".split()


def article(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word  # an apple、a ball


def some(n: int, thing: str) -> str:
    return f"{NUMBERS[n]} {thing}" + ("" if n == 1 else "s")  # one apple、three apples


def make_add(rng: random.Random) -> Task:
    name, thing, a, b = rng.choice(list(NAMES)), rng.choice(THINGS), rng.randint(1, 5), rng.randint(1, 5)
    question = f"{name} has {some(a, thing)}. {NAMES[name]} finds {NUMBERS[b]} more. How many {thing}s does {name} have now?"
    return Task("add", question, NUMBERS[a + b])


def make_subtract(rng: random.Random) -> Task:
    name, other = rng.sample(list(NAMES), 2)
    thing, a = rng.choice(THINGS), rng.randint(2, 10)
    b = rng.randint(1, a - 1)
    question = f"{name} has {some(a, thing)}. {NAMES[name]} gives {NUMBERS[b]} to {other}. How many {thing}s does {name} have left?"
    return Task("subtract", question, NUMBERS[a - b])


def make_count(rng: random.Random) -> Task:
    name, k = rng.choice(list(NAMES)), rng.randint(2, 5)
    items = [article(thing) for thing in rng.sample(THINGS, k)]
    question = f"{name} has {', '.join(items[:-1])} and {items[-1]}. How many things does {name} have?"
    return Task("count", question, NUMBERS[k])


def make_color(rng: random.Random) -> Task:
    name, (first, second), (c1, c2) = rng.choice(list(NAMES)), rng.sample(THINGS, 2), rng.sample(COLORS, 2)
    asked = rng.randrange(2)
    question = f"{name} has {article(c1)} {first} and {article(c2)} {second}. What color is the {(first, second)[asked]}?"
    return Task("color", question, (c1, c2)[asked])


def make_who(rng: random.Random) -> Task:
    people, things = rng.sample(list(NAMES), 3), rng.sample(THINGS, 3)
    asked = rng.randrange(3)
    question = f"{people[0]} has {article(things[0])}, {people[1]} has {article(things[1])} and {people[2]} has {article(things[2])}. Who has the {things[asked]}?"
    return Task("who", question, people[asked])
# ANCHOR_END: task_makers


# ANCHOR: task_api
KINDS = {"add": make_add, "subtract": make_subtract, "count": make_count, "color": make_color, "who": make_who}


def make_tasks(n: int, seed: int = 0, split: str = "train", kinds: tuple[str, ...] = tuple(KINDS)) -> list[Task]:
    """依固定種子產生 n 題，kinds 裡的題型隨機混（預設五種都有）。每一題的答案都是「從題目的數字或名字直接算出來」的，所以可以自動檢查。

    split 用題目文字的 crc32 決定：除以 10 餘 0 的題目是 "valid"，其他是 "train"。同一題不管用哪個種子產生，永遠在同一邊，
    所以 train 與 valid 不會有一模一樣的題目（第 12 章第 5 集的污染問題）。
    """
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        task = KINDS[rng.choice(kinds)](rng)
        if (zlib.crc32(task.question.encode()) % 10 == 0) == (split == "valid"):
            tasks.append(task)
    return tasks


def task_messages(task: Task, with_answer: bool = False) -> list[dict]:
    """題目轉成訊息列表；with_answer=True 時多接一則助理的回覆（SFT 的訓練資料）。"""
    messages = [{"role": "user", "content": task.question}]
    return messages + [{"role": "assistant", "content": task.answer}] if with_answer else messages


def reward(task: Task, reply: str) -> float:
    """答對 1.0、答錯 0.0。第 12 章的 exact_match：小寫、去標點與冠詞再比，所以 " Five." 也算對。"""
    return float(exact_match(reply, task.answer))
# ANCHOR_END: task_api


# ANCHOR: replies
def generate_replies(model: nn.Module, tokenizer, prompts: list[list[dict]], max_new_tokens: int = 8, temperature: float = 0.0) -> list[str]:
    """每個 prompt（訊息列表）一則回覆，到 <|endoftext|> 為止（不含它）；一直沒寫出結尾就收到 max_new_tokens 個 token。
    第 11 章的 generate_cached 要求一個 batch 的 prompt 一樣長，所以長度相同的 prompt 放在一起生成，最後照原本的順序排回去。
    temperature 大於 0 就是取樣：同一個 prompt 重複放 G 次，就是一組 G 個回答（第 15 章會用）。"""
    device = next(model.parameters()).device
    end_id = tokenizer.special_to_id[END]
    encoded = [encode_chat(tokenizer, prompt, generation_prompt=True)[0] for prompt in prompts]
    by_length: dict[int, list[int]] = {}
    for i, ids in enumerate(encoded):
        by_length.setdefault(len(ids), []).append(i)
    replies = [""] * len(prompts)
    for length, rows in by_length.items():
        out = generate_cached(model, torch.tensor([encoded[i] for i in rows], device=device), max_new_tokens, temperature)
        for i, new in zip(rows, out[:, length:].tolist()):
            replies[i] = tokenizer.decode(new[: new.index(end_id)] if end_id in new else new)
    return replies


def accuracy(model: nn.Module, tokenizer, tasks: list[Task], max_new_tokens: int = 8) -> float:
    """greedy 回答每一題，平均獎勵就是答對的比例。"""
    replies = generate_replies(model, tokenizer, [task_messages(task) for task in tasks], max_new_tokens)
    return sum(reward(task, reply) for task, reply in zip(tasks, replies)) / len(tasks)
# ANCHOR_END: replies


# ANCHOR: train_sft
@dataclass
class SFTConfig:
    base_path: str = "runs/base/model.pt"  # 從第 5 章的標準模型開始；空字串 = 不載入，隨機初始化（smoke 用）
    tokenizer_path: str = "data/tokenizer.json"
    out_dir: str = "runs/sft"
    model: ModelConfig = field(default_factory=ModelConfig)
    n_train: int = 20000  # 訓練題數
    n_valid: int = 200
    batch_size: int = 32
    max_steps: int = 1000
    lr_max: float = 1e-4  # 微調的學習率比預訓練（1e-3）小得多
    lr_min: float = 1e-5
    warmup_steps: int = 20
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 250
    mask: bool = True  # False：整串都算 loss（第 2 集的對照實驗）
    seed: int = 0
    device: str = ""  # 空字串代表用 get_device()


def train_sft(cfg: SFTConfig, tokenizer=None) -> list[dict]:
    """SFT 主迴圈：和第 5 章的 train 一樣的五步，差在資料是 (ids, mask) 而不是隨機切的文字，loss 換成 sft_loss。回傳 log 紀錄。"""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device) if cfg.device else get_device()
    os.makedirs(cfg.out_dir, exist_ok=True)
    tokenizer = tokenizer or Tokenizer.load(cfg.tokenizer_path)
    train_tasks = make_tasks(cfg.n_train, seed=cfg.seed, split="train")
    valid_tasks = make_tasks(cfg.n_valid, seed=cfg.seed + 1, split="valid")
    train_set = [encode_chat(tokenizer, task_messages(t, with_answer=True)) for t in train_tasks]
    if not cfg.mask:
        train_set = [(ids, [True] * len(ids)) for ids, _ in train_set]
    valid_ids, valid_mask = pad_batch([encode_chat(tokenizer, task_messages(t, with_answer=True)) for t in valid_tasks])

    model = TransformerLM(cfg.model).to(device)
    if cfg.base_path:
        model.load_state_dict(torch.load(cfg.base_path, map_location=device))
    optimizer = AdamW(model.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    records, log_file, t0 = [], open(os.path.join(cfg.out_dir, "log.jsonl"), "w"), time.time()
    for step in range(1, cfg.max_steps + 1):
        lr = lr_cosine_schedule(step, cfg.lr_max, cfg.lr_min, cfg.warmup_steps, cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        ids, mask = pad_batch(rng.sample(train_set, cfg.batch_size))
        loss = sft_loss(model, ids.to(device), mask.to(device))
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": lr, "grad_norm": round(grad_norm.item(), 3), "time": round(time.time() - t0, 1)}
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                model.eval()
                with torch.no_grad():  # 驗證 loss 一律只算回覆，不管訓練時有沒有套 mask，數字才能比
                    rec["val_loss"] = round(sft_loss(model, valid_ids.to(device), valid_mask.to(device)).item(), 4)
                rec["val_accuracy"] = accuracy(model, tokenizer, valid_tasks)
                model.train()
            records.append(rec)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            print(rec)
    log_file.close()
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return records
# ANCHOR_END: train_sft
