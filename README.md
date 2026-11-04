# CNN vs ResNet on CIFAR-10

## 前言

當我學完 CNN 和 Transformer 時，突然想到：Transformer 裡的殘差連接（Residual Connection）如果用在 CNN 上，會發生什麼事？剛好看見有人在做這樣的研究，於是決定自己動手實驗一次。

後來才發現，順序其實是反過來的：殘差連接最早就是 2015 年的 ResNet 為 CNN 提出的，2017 年的 Transformer 才沿用了這個設計。

## 專案簡介

在 CIFAR-10 影像分類（Image Classification）資料集上，比較卷積神經網路（Convolutional Neural Network, CNN）與殘差網路（Residual Network, ResNet）的準確率（Accuracy）、延遲（Latency）、吞吐量（Throughput）、模型大小（Model Size）與訓練成本（Training Cost）。

> **目前狀態**：程式已完成，並用 2 個訓練週期（Epoch）驗證過整條流程；正式實驗尚未跑完，所以本文件還沒有實驗結果。

## 專案結構

```text
cnn-vs-resnet/
├── README.md              ← 你在這裡
├── education.md           ← 教材：用這個專案學 CNN 與 ResNet
├── pyproject.toml         ← uv 專案設定與依賴套件（Dependencies）
├── uv.lock                ← 鎖定的依賴版本
├── .python-version        ← Python 3.12
│
├── code/                  # 所有程式碼
│   ├── cnn.py             #   BasicCNN（經典 CNN）、PlainCNN（無跳躍連接的對照組）
│   ├── resnet.py          #   ResNet-20（CIFAR 版殘差網路）
│   ├── data.py            #   CIFAR-10 載入、資料切分、資料增強
│   ├── train.py           #   訓練單一模型，保留驗證集最佳的檢查點
│   ├── evalute.py         #   測試集評估，以及延遲、吞吐量、模型大小量測
│   └── plot.py            #   從結果畫圖
│
├── tests/                 # 測試
│   └── test_setup.py      #   模型結構、資料切分、資料增強的正確性
│
│   # 以下由程式產生，不是原始碼
├── data/                  # CIFAR-10 資料集（自動下載，不進版控）
├── model/                 # 訓練好的模型權重 .pth（不進版控）
├── image/                 # 輸出的圖表
└── results/               # 實驗結果
    ├── runs/              #   每次訓練的逐週期紀錄（JSON）
    └── metrics.json       #   測試集與速度的彙整數據
```

## 資料集

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 共 60,000 張 32×32 彩色影像，分成 10 個類別（Class）：飛機（airplane）、汽車（automobile）、鳥（bird）、貓（cat）、鹿（deer）、狗（dog）、青蛙（frog）、馬（horse）、船（ship）、卡車（truck）。

| 資料切分 | 張數 | 用途 |
|---|---|---|
| 訓練集（Training Set） | 45,000 | 更新模型權重 |
| 驗證集（Validation Set） | 5,000 | 挑選最佳檢查點 |
| 測試集（Test Set） | 10,000 | 只用於最終評估 |

原始訓練資料有 50,000 張，用固定的隨機種子（Random Seed）切出驗證集，所以每次實驗的切分都相同。

## 比較的模型

| 指令名稱 | 模型 | 參數量（Parameters） | 模型大小 | 運算量（MACs） |
|---|---|---:|---:|---:|
| `cnn` | BasicCNN | 4,293,194 | 16.38 MB | 14.5 M |
| `plain` | PlainCNN-20 | 269,722 | 1.03 MB | 40.6 M |
| `resnet` | ResNet-20 | 269,722 | 1.03 MB | 40.6 M |

- **BasicCNN**：3 層卷積層（Convolutional Layer）、2 層最大池化（Max Pooling）、2 層全連接層（Fully Connected Layer），沒有批次正規化（Batch Normalization, BN）。
- **PlainCNN-20**：結構與 ResNet-20 相同，只拿掉跳躍連接（Skip Connection）。降採樣（Downsampling）用步幅（Stride）為 2 的卷積，不用池化。
- **ResNet-20**：20 層殘差網路，架構來自 [He et al. 2015](https://arxiv.org/abs/1512.03385) 第 4.2 節。捷徑（Shortcut）採用無參數的 option A，尺寸改變時隔點取樣並以 0 補足通道（Channel）。

### 怎麼解讀比較結果

- **`cnn` vs `resnet`**：經典 CNN 與 ResNet 的整體比較。兩者的深度、參數量和學習率都不同，所以差距不能只歸因於某一項設計。
- **`plain` vs `resnet`**：參數量、運算量和初始權重都相同，唯一的差別是跳躍連接，可以單獨看它的效果。

## 實驗設定

| 項目 | 設定 |
|---|---|
| 資料增強（Data Augmentation） | 只用於訓練集：隨機裁切（Random Crop，padding 4）、隨機水平翻轉（Random Horizontal Flip） |
| 標準化（Normalization） | 所有資料都依 CIFAR-10 的平均值與標準差處理 |
| 損失函數（Loss Function） | 交叉熵（Cross Entropy） |
| 優化器（Optimizer） | 隨機梯度下降（Stochastic Gradient Descent, SGD），動量（Momentum）0.9 |
| 權重衰減（Weight Decay） | 1e-4 |
| 學習率（Learning Rate） | `plain`、`resnet` 為 0.1；`cnn` 為 0.01，因為沒有 BN，0.1 容易發散（Diverge） |
| 學習率排程（LR Scheduler） | 餘弦退火（Cosine Annealing） |
| 批次大小（Batch Size） | 128 |
| 訓練週期（Epoch） | 60 |
| 隨機種子 | 0、1、2，用來觀察結果的波動 |

選擇檢查點時只看驗證集，避免測試集資訊洩漏（Data Leakage）而高估準確率。

## 環境

- 套件管理：[uv](https://docs.astral.sh/uv/)
- Python 3.12、PyTorch 2.15、torchvision 0.29
- 開發機器：Mac mini（Apple M4 Pro、24 GB 記憶體），使用 Metal Performance Shaders（MPS）做 GPU 加速

沒有 MPS 的機器會自動改用 CPU，訓練會慢很多。

## 使用方式

### 1. 安裝與測試

```bash
uv sync           # 安裝依賴套件
uv run pytest     # 執行測試
```

### 2. 訓練

訓練單一模型（第一次執行會自動下載約 170 MB 的 CIFAR-10）：

```bash
uv run python code/train.py --model resnet --seed 0
```

| 參數 | 說明 | 預設值 |
|---|---|---|
| `--model` | `cnn`、`plain` 或 `resnet` | 必填 |
| `--epochs` | 訓練週期數 | 60 |
| `--seed` | 隨機種子 | 0 |

完整實驗是 3 個模型 × 3 個種子，在開發機器上估計約 42 分鐘：

```bash
for seed in 0 1 2; do
  for model in cnn plain resnet; do
    uv run python code/train.py --model $model --seed $seed
  done
done
```

### 3. 評估與畫圖

```bash
uv run python code/evalute.py    # 產生 results/metrics.json
uv run python code/plot.py       # 產生 image/ 裡的圖表
```

這三個步驟要依序執行：`train.py` 產生檢查點，`evalute.py` 再產生 `results/metrics.json`，最後 `plot.py` 才畫得出圖。執行 `evalute.py` 時請讓機器保持閒置，否則延遲數字會失真。

## 比較指標

| 指標 | 量測方式 |
|---|---|
| 準確率（Accuracy） | 測試集整體與各類別，取 3 個種子的平均值與標準差（Standard Deviation） |
| 延遲（Latency） | 批次大小 1，報告中位數（p50）與第 95 百分位數（p95），單位毫秒（ms），MPS 與 CPU 分開量 |
| 吞吐量（Throughput） | 批次大小 256，每秒處理的影像數（images/s），MPS 與 CPU 分開量 |
| 模型大小（Model Size） | 參數量，以及 float32 權重所占容量（MB） |
| 運算量（Computation） | 單張影像前向傳播（Forward Pass）的乘加運算次數（Multiply-Accumulate Operations, MACs） |
| 訓練成本（Training Cost） | 每個週期的秒數（含資料載入，不含第一個週期）與總訓練時間 |

量測速度前會先暖機（Warm-up），並且讓各模型輪流量測，避免機器升溫降頻（Thermal Throttling）只影響到其中一個模型。每次計時都會等 GPU 真正算完才停表。

## 輸出的圖表

| 檔案 | 內容 |
|---|---|
| `accuracy.png` | 各模型的測試集準確率，含每個種子的結果 |
| `speed.png` | 延遲與吞吐量，MPS 和 CPU 分開呈現 |
| `curves.png` | 訓練損失、驗證損失與驗證準確率的學習曲線（Learning Curve） |
| `per_class.png` | 各類別的測試集準確率 |

## 實驗結果

尚未完成，正式實驗跑完後會補上。

## 常見問題

**`plot.py` 顯示找不到 `metrics.json`？**
要先訓練模型，再執行 `evalute.py`。

**CIFAR-10 下載很久沒有進度？**
下載沒有逾時機制，連線中斷時可能一直等待。可以停止程式後用 `curl` 續傳，完成後再重新執行訓練：

```bash
curl -L -C - -o data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

## 參考資料

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Krizhevsky, A. (2009). [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).

