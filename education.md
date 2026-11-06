# 從這個專案學 CNN 與 ResNet

這份教材寫給學過機器學習的夥伴們。你應該已經要知道梯度下降（Gradient Descent）、反向傳播（Backpropagation）、損失函數（Loss Function）和過擬合（Overfitting）等等 ML 基礎，所以這裡不重講這些，而是專注在這個專案的三件事：

1. 模型為什麼長這樣，數字怎麼算出來。
2. 殘差連接（Residual Connection）到底解決什麼問題。
3. 怎麼設計一個公平的實驗，並正確量測速度。

建議一邊讀一邊打開 `code/` 裡對應的檔案。

> 本文的參數量與運算量都是實際模型算出來的。文中沒有準確率和延遲數字，因為正式實驗還沒跑完；提到實驗會怎樣時，都是根據理論的預期，請以你實際跑出的結果為準。

## 1. 這個專案想回答什麼

| 問題 | 比較對象 |
|---|---|
| 經典 CNN 和 ResNet 整體差多少？ | BasicCNN vs ResNet-20 |
| 跳躍連接（Skip Connection）本身帶來多少效果？ | PlainCNN-20 vs ResNet-20 |

第一個問題比較的是兩套完整設計，第二個問題才是控制變因的實驗。第 7 節會解釋為什麼這個區分很重要。

## 2. 資料流程（`code/data.py`）

CIFAR-10 有 60,000 張 32×32 彩色影像，共 10 類，每張影像的張量形狀是 `(3, 32, 32)`。

### 為什麼要切三份

| 資料切分 | 張數 | 誰會看到它 |
|---|---|---|
| 訓練集（Training Set） | 45,000 | 梯度更新 |
| 驗證集（Validation Set） | 5,000 | 挑選檢查點（Checkpoint） |
| 測試集（Test Set） | 10,000 | 最後評估一次 |

「挑選最佳檢查點」本身也是一種擬合。如果用測試集來挑，等於讓測試集影響了最終模型，回報的準確率就會偏高，這叫資料洩漏（Data Leakage）。所以 `train.py` 完全不碰測試集。

### 資料增強（Data Augmentation）

```python
transforms.RandomCrop(32, padding=4)      # 四周補 4 格，再隨機裁回 32×32
transforms.RandomHorizontalFlip()         # 50% 機率左右翻轉
```

這相當於告訴模型：物體平移幾個像素或左右相反，類別都不變。每個訓練週期（Epoch）看到的影像都略有不同，能降低過擬合。

有三個細節值得注意：

- 驗證集和測試集不做隨機增強，否則評估結果每次都不同，也不代表真實表現。
- 不是所有增強都合理。垂直翻轉對 CIFAR-10 就不太自然，因為現實中很少有倒過來的船。
- 訓練和驗證要用不同的 `transform`，所以 `data.py` 把同一份資料載入兩次，再用 `Subset` 取不同的索引。

## 3. 三個模型（`code/cnn.py`、`code/resnet.py`）

### 卷積輸出尺寸

讀模型時最常用的公式：

```text
輸出尺寸 = floor((輸入尺寸 + 2 × padding - kernel) / stride) + 1
```

`kernel=3, padding=1, stride=1` 時：(32 + 2 - 3) / 1 + 1 = 32，尺寸不變。
`kernel=3, padding=1, stride=2` 時：floor(31 / 2) + 1 = 16，尺寸減半。

### BasicCNN

```text
(3, 32, 32)
  → conv1 + ReLU    → (32, 32, 32)
  → max pool        → (32, 16, 16)
  → conv2 + ReLU    → (64, 16, 16)
  → max pool        → (64, 8, 8)
  → conv3 + ReLU    → (128, 8, 8)
  → flatten         → (8192,)
  → fc1 + ReLU      → (512,)
  → fc2             → (10,)
```

這是教科書式的設計：空間尺寸逐漸縮小，通道（Channel）逐漸增加，最後交給全連接層（Fully Connected Layer）分類。

### ResNet-20

```text
(3, 32, 32)
  → conv1 + BN + ReLU                → (16, 32, 32)
  → layer1：3 個殘差區塊              → (16, 32, 32)
  → layer2：3 個殘差區塊（第 1 個 stride=2） → (32, 16, 16)
  → layer3：3 個殘差區塊（第 1 個 stride=2） → (64, 8, 8)
  → 全域平均池化                      → (64,)
  → fc                               → (10,)
```

名稱裡的 20 是有權重的層數：1 層開頭卷積 + 9 個區塊 × 2 層卷積 + 1 層全連接 = 20。

### 殘差區塊（Residual Block）

```text
        x ─────────────────────┐
        │                      │  捷徑（Shortcut）
   conv → BN → ReLU            │
        │                      │
   conv → BN                   │
        │                      │
        └──────── + ───────────┘
                  │
                ReLU
```

對應的程式只有關鍵一行：

```python
out = torch.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
out += self.shortcut(x)  # Skip connection
out = torch.relu(out)
```

### 捷徑遇到尺寸不合怎麼辦

相加時兩邊形狀必須相同。layer2 的第一個區塊把 `(16, 32, 32)` 變成 `(32, 16, 16)`，`x` 就不能直接加。常見做法有兩種：

| 做法 | 方式 | 參數 |
|---|---|---|
| Option A（本專案） | 隔點取樣縮小，通道不足補 0 | 無 |
| Option B | 用 1×1 卷積轉換 | 有 |

本專案選 option A，因為它不增加參數，才能讓 ResNet-20 和 PlainCNN-20 的參數量完全相同。

```python
x = x[:, :, ::self.stride, ::self.stride]        # 空間尺寸減半
half = self.extra_channels // 2
return F.pad(x, (0, 0, 0, 0, half, half))        # 通道前後各補 half 個 0
```

`F.pad` 的參數從最後一維往前算，每兩個數字一組：寬度、高度、通道。

### 為什麼卷積設定 `bias=False`

ResNet 的卷積後面都接批次正規化（Batch Normalization, BN）。BN 會先減掉平均值，卷積的偏差（Bias）在這一步就被抵消了，然後 BN 再用自己的可學習參數 β 做平移，所以卷積的 bias 是多餘的。

BasicCNN 沒有 BN，所以保留 bias。

## 4. 參數量與運算量怎麼算

### 公式

設卷積核大小 k，輸入通道 C_in，輸出通道 C_out，輸出特徵圖大小 H × W：

```text
卷積參數量 = (k × k × C_in + 1) × C_out      （沒有 bias 就不加 1）
卷積 MACs  = k × k × C_in × C_out × H × W
全連接參數量 = (輸入 + 1) × 輸出
全連接 MACs  = 輸入 × 輸出
```

MACs 是乘加運算次數（Multiply-Accumulate Operations），一次乘法加一次加法算 1 次。有些工具會算成 2 FLOPs，所以 `evalute.py` 把 FLOPs 除以 2。

關鍵差別是卷積的 MACs 多乘了 H × W：同一組權重會在每個位置重複使用，這就是權重共享（Weight Sharing）。

### BasicCNN

| 層 | 參數量 | MACs | 每個權重用幾次 |
|---|---:|---:|---:|
| `conv1` | 896 | 884,736 | 1,024 |
| `conv2` | 18,496 | 4,718,592 | 256 |
| `conv3` | 73,856 | 4,718,592 | 64 |
| `fc1` | 4,194,816 | 4,194,304 | 1 |
| `fc2` | 5,130 | 5,120 | 1 |
| **合計** | **4,293,194** | **14,521,344** | |

`fc1` 占全部參數的 97.7%，運算量卻只占 28.9%。

### ResNet-20

| 部分 | 參數量 | MACs |
|---|---:|---:|
| 開頭卷積 | 432 | 442,368 |
| layer1 卷積 | 13,824 | 14,155,776 |
| layer2 卷積 | 50,688 | 12,976,128 |
| layer3 卷積 | 202,752 | 12,976,128 |
| BN | 1,376 | 不計 |
| `fc` | 650 | 640 |
| **合計** | **269,722** | **40,551,040** |

把兩張表放在一起，就能看出一個反直覺的結果：

| | BasicCNN | ResNet-20 |
|---|---:|---:|
| 參數量 | 429 萬 | 27 萬 |
| MACs | 1,452 萬 | 4,055 萬 |

BasicCNN 參數多約 16 倍，運算量卻不到 ResNet-20 的四成。**參數量是存了幾個數字，運算量是這些數字被用了幾次**，兩者不是同一件事。

模型大小則直接由參數量決定：float32 每個參數 4 bytes，所以 4,293,194 × 4 ÷ 1024² = 16.38 MB。

### flatten 與全域平均池化

ResNet-20 參數少的主因之一，是用全域平均池化（Global Average Pooling, GAP）取代 `flatten`。

| | `flatten` | GAP |
|---|---|---|
| 輸出 | 保留每個位置 | 每個通道只剩一個平均值 |
| 參數量 | 多 | 少 |
| 位置資訊 | 保留 | 丟失 |
| 輸入尺寸 | 必須固定 | 可以變動 |

注意：**跳躍連接不會減少參數**。PlainCNN-20 沒有跳躍連接，參數量同樣是 269,722。

## 5. 殘差連接為什麼有用

### 退化問題（Degradation Problem）

直覺上網路越深應該越強：多出來的層只要學成恆等映射（Identity Mapping），表現至少不會變差。但 ResNet 論文發現，56 層的普通網路比 20 層的**訓練誤差更高**。

這點很重要：訓練誤差變高代表問題不是過擬合，因為過擬合會是訓練誤差低、測試誤差高。問題出在優化（Optimization）：解確實存在，但優化器找不到。

### 改學殘差

設希望學到的映射是 H(x)。普通網路直接學 H(x)，殘差區塊改學：

```text
F(x) = H(x) - x
輸出 = F(x) + x
```

如果最佳解接近恆等映射，普通網路得讓多層非線性運算精準地組合出恆等映射，而殘差區塊只要把 F(x) 推向 0，這容易得多。

### 從梯度來看

設 y = x + F(x)，反向傳播時：

```text
∂L/∂x = ∂L/∂y × (I + ∂F/∂x)
```

那個 I 表示梯度有一條不經過權重矩陣的路徑。普通網路的梯度是一連串 ∂F/∂x 連乘，容易變得很小。殘差網路展開後則有多條路徑，其中一些很短，前面的層比較容易收到有效的訊號。

### 和 Transformer 的關係

Transformer 的每個子層都有 `x + Sublayer(x)`，這個設計來自 ResNet。ResNet 發表於 2015 年，Transformer 發表於 2017 年。兩者的動機相同：沒有殘差連接，很深的網路很難訓練。

### 為什麼 20 層可能看不出大差距

退化問題要夠深才明顯。在 20 層時，BN 加上良好的初始化，已經足以讓普通網路訓練得動，所以 PlainCNN-20 和 ResNet-20 的差距可能不大。想看到更明顯的差距，要加深到 56 層以上。

## 6. 訓練程式（`code/train.py`）

### 一個訓練步驟

```python
optimizer.zero_grad()              # PyTorch 預設累加梯度，要先清空
outputs = model(inputs)            # 前向傳播
loss = criterion(outputs, labels)  # 計算 loss
loss.backward()                    # 反向傳播
optimizer.step()                   # 更新權重
```

`nn.CrossEntropyLoss` 內部已經包含 softmax，所以模型直接輸出原始分數（Logits），不用自己加 softmax。

### `model.train()` 和 `model.eval()`

這不是裝飾，對 BN 有實際影響：

| 模式 | BN 使用的統計量 |
|---|---|
| `train()` | 當前批次的平均值與變異數，並更新移動平均 |
| `eval()` | 訓練時累積的移動平均 |

忘記切換到 `eval()`，驗證結果會受批次內容影響，也會污染統計量。`torch.no_grad()` 是另一回事：它只是不記錄計算圖，用來省記憶體和時間。

### 超參數（Hyperparameter）

| 設定 | 值 | 理由 |
|---|---|---|
| 優化器 | SGD + 動量（Momentum）0.9 | ResNet 論文的標準設定 |
| 權重衰減（Weight Decay） | 1e-4 | 抑制權重過大 |
| 學習率 | `plain`、`resnet` 為 0.1，`cnn` 為 0.01 | BasicCNN 沒有 BN，較大的學習率容易發散 |
| 學習率排程 | 餘弦退火（Cosine Annealing） | 前期探索、後期細調 |

餘弦退火的公式（本專案最低學習率為 0）：

```text
lr(t) = lr₀ × (1 + cos(π × t / T)) / 2
```

高學習率時，驗證準確率常常跳動得很厲害；學習率降下來後才會大幅上升。所以訓練中途的數字不能拿來比較，要看訓練結束的結果。

### Kaiming 初始化

```python
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

ReLU 會把約一半的值變成 0，Kaiming 初始化（Kaiming Initialization）依此調整權重的變異數，讓訊號經過多層後不至於一路縮小或放大。

## 7. 實驗設計：怎樣才算公平

### 控制變因（Controlled Variable）

PlainCNN-20 和 ResNet-20 的比較中，以下全部相同：

- 層數、通道數、參數量、運算量
- 同一個種子下的初始權重（`tests/test_setup.py` 有檢查）
- 資料切分、資料增強、超參數

唯一的差別是跳躍連接，所以結果的差距才能歸因於它。

BasicCNN vs ResNet-20 就不同了：深度、參數量、BN、池化方式、學習率全都不一樣。這個比較有實用價值，但不能下「ResNet 贏是因為跳躍連接」這種結論。

### 多個隨機種子（Random Seed）

只跑一次的結果可能來自運氣：初始權重、資料順序和增強都有隨機性。本專案跑 3 個種子，回報平均值與標準差。

判斷差距時可以先問：差距有沒有明顯大於標準差？如果兩個模型只差 0.3%，而標準差也是 0.3%，就不該說誰比較好。3 個種子的樣本仍然很小，所以這只是初步判斷，不是嚴格的統計檢定。

### 超參數公平嗎

本專案沒有替每個模型仔細調參，BasicCNN 的學習率只是根據它沒有 BN 而調低。嚴謹的比較應該對每個模型做同等的超參數搜尋。解讀 BasicCNN 的結果時要記得這個限制。

## 8. 速度量測（`code/evalute.py`）

### 指標

| 指標 | 意義 | 適用情境 |
|---|---|---|
| 延遲（Latency） | 處理一張影像要多久 | 即時服務 |
| 吞吐量（Throughput） | 每秒能處理幾張影像 | 批次處理 |

GPU 的吞吐量通常遠高於單張處理的倒數，因為大批次能填滿平行運算單元。

### 為什麼要用中位數和 p95

延遲的分布常有長尾，偶爾一次系統排程就可能拖慢很多。平均值容易被極端值影響，所以用中位數（p50）表示典型情況，用第 95 百分位數（p95）表示「慢的時候有多慢」。

### 量測的三個陷阱

**1. GPU 是非同步的。** `model(x)` 回傳時，GPU 可能還沒算完，只是工作排進佇列了。不同步就停表，量到的只是送出指令的時間：

```python
synchronize(device)
start_time = time.perf_counter()
model(inputs)
synchronize(device)
```

**2. 第一次執行特別慢。** 開頭要做核心編譯和記憶體配置，所以要先暖機（Warm-up），也不把第一個週期算進每週期秒數。

**3. 機器會升溫降頻。** 如果先量完 A 再量 B，後量的模型可能吃虧，所以各模型輪流量測，同時也要保持機器閒置。

### 運算量不等於延遲

MACs 只計算數學運算，實際時間還受這些因素影響：

- **層數**：每層都有啟動開銷。批次大小為 1 時運算很小，開銷可能占大部分，所以層數多的模型可能比較吃虧。
- **記憶體存取**：BasicCNN 的 `fc1` 要讀取 16 MB 左右的權重，卻只做相對少的運算。
- **硬體與實作**：卷積通常優化得很好，同樣的 MACs 不一定花同樣的時間。

理論上 ResNet-20 會比 PlainCNN-20 略慢：MACs 一樣，但多了相加運算，而且要保留 `x` 到區塊結束。實際差多少，要看你量出來的數字。

## 9. 動手做一次

```bash
uv sync
uv run pytest

# 先用少量週期確認流程
uv run python code/train.py --model resnet --epochs 2 --seed 99
uv run python code/evalute.py
uv run python code/plot.py
```

確認無誤後，刪掉 `seed99` 的檔案，再依照 `README.md` 跑完整實驗。否則 `evalute.py` 會把它們算進平均。

看結果時建議依序問自己：

1. **學習曲線**：訓練與驗證的差距有多大？哪個模型過擬合最嚴重？
2. **準確率**：差距有沒有明顯大於種子之間的波動？
3. **各類別準確率**：哪些類別最難？三個模型會在同樣的類別上犯錯嗎？（貓和狗通常是 CIFAR-10 的難點）
4. **速度**：MPS 和 CPU 的排名一樣嗎？運算量的排名能預測延遲的排名嗎？

## 10. 練習題

**概念題**

1. 一層 `Conv2d(64, 128, kernel_size=3, padding=1)`，有 bias，輸入 `(64, 16, 16)`。它的參數量和 MACs 各是多少？
2. 為什麼「56 層普通網路的訓練誤差比 20 層高」可以排除過擬合？
3. 如果把 `model.eval()` 拿掉，評估結果會受到什麼影響？

**實作題**

4. 把 BasicCNN 的 `flatten` 換成 `nn.AdaptiveAvgPool2d((1, 1))`。`fc1` 的輸入要改成多少？參數量變多少？準確率如何變化？
5. 把 ResNet 改成 option B（1×1 卷積捷徑）。參數量增加多少？為什麼這樣就不再是嚴格的控制變因實驗？
6. 把 PlainCNN 和 ResNet 都加深到 56 層（每個 stage 9 個區塊），比較訓練誤差，看你能不能重現退化問題。
7. 拿掉資料增強再訓練一次，比較訓練與驗證準確率的差距。
8. 把 BasicCNN 的學習率改成 0.1，觀察是否真的會發散。

**第 1 題答案**：參數量 (3 × 3 × 64 + 1) × 128 = 73,856。MACs 為 3 × 3 × 64 × 128 × 16 × 16 = 18,874,368。

## 延伸閱讀

- He et al. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- He et al. (2016). [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- Ioffe & Szegedy (2015). [Batch Normalization](https://arxiv.org/abs/1502.03167)
- He et al. (2015). [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)（Kaiming 初始化）
- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
