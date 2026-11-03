import torch

from llmz.dist import BucketedDDP, ShardedOptimizer, all_reduce_gradients, broadcast_parameters, launch, ring_all_reduce
from llmz.functional import cross_entropy
from llmz.model import ModelConfig, TransformerLM
from llmz.optim import AdamW

CFG = ModelConfig(vocab_size=50, context_length=8, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def _loss(model, ids):
    return cross_entropy(model(ids[:, :-1]).reshape(-1, 50), ids[:, 1:].reshape(-1))


def _ring_worker(rank, world):
    x = torch.arange(10.0) * (rank + 1)  # 10 不能被 3 整除，三塊大小不同
    expected = torch.arange(10.0) * sum(range(1, world + 1))
    assert torch.equal(ring_all_reduce(x), expected)


def test_ring_all_reduce_three_ranks():
    launch(_ring_worker, 3)


def _ddp_worker(rank, world, mode):
    torch.manual_seed(1)
    batches = [torch.randint(0, 50, (4, 9)) for _ in range(2)]
    torch.manual_seed(10)
    reference = TransformerLM(CFG)  # 單一 process 看整個 batch
    torch.manual_seed(rank + 10)  # 故意讓每個 rank 的初始參數不同，要靠 broadcast 統一
    model = TransformerLM(CFG)
    if mode == "bucketed":
        ddp = BucketedDDP(model, bucket_mb=0.01)  # bucket 很小，才會分成好幾個
        assert len(ddp.buckets) > 1
    else:
        broadcast_parameters(model)
    for ids in batches:  # 跑兩次，確認 bucket 的計數有歸零
        _loss(model, ids[rank * 2 : rank * 2 + 2]).backward()  # 每個 rank 只看自己那一半
        ddp.finish() if mode == "bucketed" else all_reduce_gradients(model)
        _loss(reference, ids).backward()
        for p, q in zip(model.parameters(), reference.parameters()):
            assert torch.allclose(p.grad, q.grad, atol=1e-6)
            p.grad = q.grad = None


def test_naive_ddp_matches_full_batch_gradients():
    launch(_ddp_worker, 2, "naive")


def test_bucketed_ddp_matches_full_batch_gradients():
    launch(_ddp_worker, 2, "bucketed")


def _sharded_worker(rank, world):
    torch.manual_seed(1)
    batches = [torch.randint(0, 50, (4, 9)) for _ in range(3)]
    models = []
    for _ in range(2):
        torch.manual_seed(10)
        models.append(TransformerLM(CFG))
    sharded = ShardedOptimizer(models[0].parameters(), AdamW, lr=1e-2)
    plain = AdamW(models[1].parameters(), lr=1e-2)
    for ids in batches:
        for model, opt in zip(models, (sharded, plain)):
            _loss(model, ids).backward()
            opt.step()
            opt.zero_grad()
    for p, q in zip(*(m.parameters() for m in models)):
        assert torch.allclose(p, q, atol=1e-6)
    mine = sum(p.numel() for p in sharded.optimizer.param_groups[0]["params"])
    assert mine < sum(p.numel() for p in models[0].parameters())  # 每個 rank 只管一部分


def test_sharded_optimizer_matches_plain_adamw():
    launch(_sharded_worker, 2)
