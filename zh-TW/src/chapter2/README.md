# autograd 與訓練迴圈

第 1 章我們用有限差分算梯度，每個參數都要多算一次函數，模型一大就不可能這樣做。這一章讓 PyTorch 自己算梯度（autograd），然後學會把參數裝進 `nn.Module`，寫出全書之後每一次訓練都長一樣的五步迴圈：forward、算 loss、`backward()`、`step()`、`zero_grad()`。

我們會用一個只有兩個參數的線性回歸把整條流程走完，順便學會把訓練好的參數存檔、載回，以及在動手之前先估算一個模型要多少記憶體、多少計算量。讀完這章，第 4 章的 Transformer 只是把「模型」換成比較大的東西而已。

這一章對應 CS336 Lecture 2 的 PyTorch 與資源估算部分，見 <https://cs336.stanford.edu>。
