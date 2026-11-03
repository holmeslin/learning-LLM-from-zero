# 平行化：原理與 CPU 模擬

一張卡放不下、或一張卡算太久的時候，就要把工作分給很多張卡。這一章介紹四種分法：每張卡算不同資料的 data parallel、把優化器狀態與參數切開存的 sharding（ZeRO 與 FSDP）、把一層切給好幾張卡的 tensor parallel，以及把不同層放在不同卡的 pipeline parallel。

我們沒有多張 GPU，但 PyTorch 的 `torch.distributed` 可以用 Gloo 後端在 CPU 上起好幾個 process，每個 process 假裝是一張卡，通訊的寫法和真正的多卡訓練一模一樣。我們用它從頭寫出 ring all-reduce、DDP、bucketing、optimizer state sharding，並驗證結果和單機訓練相同；tensor 與 pipeline parallel 只示範原理。最後一集學會估算通訊和計算哪個是瓶頸。

這一章對應 CS336 Assignment 2 的分散式訓練部分與平行化的講次，見 <https://cs336.stanford.edu>。
