# ANN-Bench: Approximate Nearest Neighbor Search for Speaker Embeddings

## Proposal Report

### 動機與目標

Speech Language Model 將語音編碼為高維 speaker embedding，可用於說話者辨識（Speaker Verification）——即「這段語音和資料庫中哪個說話者最像？」當資料庫規模擴大，如何快速找到最相似的 embedding 成為關鍵問題。

本專題從頭實作三種向量索引方法，在 speaker embedding 資料集上比較其效能，從演算法層面理解 Approximate Nearest Neighbor（ANN）搜尋的核心原理與取捨。

### 預期功能

1. **三種索引方法實作**
   - Flat Search：暴力搜尋，作為 baseline
   - KD-Tree：空間分割樹，適合低維度精確搜尋
   - LSH（Locality-Sensitive Hashing）：適合高維度近似搜尋

2. **效能 Benchmark**
   - 比較三種方法在不同資料量下的 recall@k 與 query latency
   - 分析高維度對各索引效能的影響

3. **互動式查詢介面（Streamlit）**
   - 選擇說話者，即時返回 Top-K 最相似結果

4. **視覺化**
   - **Three.js**：將 speaker embedding 降維後呈現為 3D 互動點雲，同說話者自然聚集，查詢時鄰居節點發光連線
   - **Manim**：KD-Tree 空間分割與查詢走訪動畫、LSH bucket 分群動畫

### 使用技術

- **語言**：Python、JavaScript
- **Embedding 來源**：預訓練 ECAPA-TDNN（SpeechBrain），直接抽取 speaker embedding，不自行訓練模型
- **資料集**：LibriSpeech train-clean-100（251 位說話者）
- **索引實作**：純 Python，不使用 FAISS 等現成 ANN 函式庫
- **視覺化**：Three.js（3D 點雲）、Manim（演算法動畫）
- **UI**：Streamlit

### 時程規劃

| 週次 | 預計進度 |
|------|---------|
| Week 7 | Proposal 繳交、建立環境、確認 LibriSpeech 資料流程 |
| Week 8 | 實作 Flat Search + benchmark 框架 |
| Week 9 | 實作 KD-Tree（建樹、查詢、剪枝） |
| Week 10 | 實作 LSH（hash function 設計、bucket 查詢） |
| Week 11 | Prototype 繳交、完成三方法 benchmark 比較 |
| Week 12 | Streamlit 查詢介面 + Three.js 3D 點雲 |
| Week 13 | Manim 演算法動畫（KD-Tree 走訪、LSH bucket） |
| Week 14 | 整合、完整測試、撰寫分析 |
| Week 15 | Final Report 繳交、錄製 demo 影片 |

### 與課程的關聯

| 資料結構 / 演算法 | 對應實作 |
|---|---|
| Binary Search Tree | KD-Tree 建樹與遞迴分割 |
| Hashing | LSH hash function 族設計與 collision 控制 |
| Tree Pruning | KD-Tree 查詢時跳過不可能包含答案的子樹 |
| Space-Time Tradeoff | 三種方法在準確度、建立時間、查詢速度間的取捨 |
| Curse of Dimensionality | 分析 KD-Tree 在高維退化的現象與成因 |

---

## Prototype Report

### 目前進度

三種索引方法與 benchmark 框架均已完成，可在合成資料上執行完整比較：

- **`src/data/loader.py`**：LibriSpeech embedding 抽取流程（ECAPA-TDNN / SpeechBrain），含合成資料 fallback
- **`src/index/flat.py`**：Flat Search，以 cosine similarity 對全資料庫暴力搜尋，作為 ground truth
- **`src/index/kdtree.py`**：KD-Tree，依最高 variance 維度遞迴切割，查詢時使用 branch-and-bound 剪枝
- **`src/index/lsh.py`**：LSH，以 random projection 設計 hash function，多組 hash table 提升 recall
- **`src/benchmark/eval.py`**：recall@k 與 query latency 測量框架
- **`src/benchmark/run.py`**：三方法完整比較，輸出 benchmark 圖表
- **`src/benchmark/scaling.py`**：不同資料量（N=100–1000）下的 recall 與 latency 變化曲線

**合成資料（192 維，50 位說話者，1000 筆）benchmark 結果：**

| 方法 | Recall@10 | Latency (ms) |
|------|-----------|--------------|
| Flat Search | 1.000 | 0.05 |
| KD-Tree | ~0.23 | 1.13 |
| LSH (n_bits=4, n_tables=16) | ~0.92 | 0.41 |

### 遇到的困難

1. **LSH 參數調校**：初始設定 `n_bits=12` 導致 recall 僅 0.13。原因是 4096 個 bucket 對 1000 筆資料過於稀疏，鄰居幾乎不會落入同一 bucket。透過系統性的參數搜尋，`n_bits=4, n_tables=16` 可將 recall 提升至 0.92。

2. **KD-Tree 高維退化**：192 維下 KD-Tree recall 僅約 0.23，符合理論預期（curse of dimensionality）。此現象反而成為本專題最有力的實驗結果，清楚說明為何高維向量搜尋需要 LSH 等近似方法。

### 下一步計畫

- ✅ Streamlit 查詢介面（`src/ui/app.py`）
- ✅ Three.js 3D speaker embedding 點雲視覺化（`src/ui/threejs_component.py`）
- ✅ Manim KD-Tree 走訪動畫（`src/viz/animation/kdtree_anim.py`）
- ✅ Manim LSH bucket 動畫（`src/viz/animation/lsh_anim.py`）
- ⬜ 安裝 SpeechBrain，使用真實 LibriSpeech 資料跑完整 benchmark
- ⬜ 錄製 demo 影片並上傳 YouTube

### 與課程的關聯

| 課程概念 | 實作對應 |
|----------|---------|
| Binary Search Tree | KD-Tree 以遞迴中位數切割建樹，結構等同多維 BST |
| Tree Pruning | 查詢時若切割超平面距離 > 當前最差鄰居距離，整棵子樹跳過 |
| Hashing & Collision | LSH 設計讓相似向量高機率產生相同 hash，collision 是刻意的 |
| Space-Time Tradeoff | 三種方法的 recall/latency 數據直接量化了準確度與速度的取捨 |
| Curse of Dimensionality | KD-Tree 在 192 維下 recall 退化至 0.23，實驗驗證理論 |

---

## Final Report

### 專案說明

ANN-Bench 是一個從頭實作三種向量索引方法的 speaker verification 搜尋引擎，搭配互動介面與演算法動畫。

**核心問題**：Speech Language Model 將語音編碼為高維 speaker embedding。給定一段查詢語音，如何在大型資料庫中快速找到最相似的說話者？

**三種方法的實作與比較：**

| 方法 | 原理 | Recall@10 | Latency |
|------|------|-----------|---------|
| Flat Search | 對全資料庫計算 cosine similarity | 1.000 | 0.05 ms |
| KD-Tree | 遞迴空間分割 + branch-and-bound 剪枝 | ~0.23 | 1.13 ms |
| LSH | Random projection hash，多組 hash table | ~0.92 | 0.41 ms |

KD-Tree 在 192 維下 recall 僅 0.23，直接呈現 curse of dimensionality；LSH 以近似換取速度，在高維空間中仍保持高 recall。

**視覺化：**
- **Three.js 3D 點雲**：speaker embedding 以 PCA 降至 3D，查詢點白色脈衝，Top-K 鄰居金色連線
- **Streamlit UI**：選擇說話者與搜尋方法，即時顯示搜尋結果與 3D 視覺化
- **Manim 動畫**：
  - KD-Tree BuildScene：分割線逐步長出
  - KD-Tree QueryScene：走訪路徑發光，剪枝區域淡出
  - LSH HashScene：hyperplane 切割，points 依 bucket 著色
  - LSH QueryScene：query 被雜湊至 bucket，只有該 bucket 的候選點亮起

### 使用方式

#### 方法一：Docker（推薦）

```bash
git clone https://github.com/nia1003/dsap-project.git
cd dsap-project

# 建立並啟動（第一次約 5 分鐘，使用合成資料，即開即用）
docker compose up --build

# 之後再跑
docker compose up
```

開啟 **http://localhost:8601**

> **使用真實 LibriSpeech 資料（選用）**：首次執行會自動下載約 1GB 資料，需額外 10–15 分鐘。
> ```bash
> # 方式一：profile
> docker compose --profile real up --build
> # 方式二：環境變數
> USE_SYNTHETIC=0 docker compose up --build
> ```

#### 方法二：本機直接執行

**安裝依賴：**
```bash
pip install -r requirements.txt
```

**啟動互動介面：**
```bash
streamlit run src/ui/app.py
```

開啟 **http://localhost:8501**

**執行 benchmark：**
```bash
python main.py benchmark --synthetic   # 合成資料，無需下載
python main.py scaling --synthetic
```

**渲染 Manim 動畫：**
```bash
# 低畫質預覽
manim -pql src/viz/animation/kdtree_anim.py QueryScene
manim -pql src/viz/animation/lsh_anim.py QueryScene

# 高畫質（demo 影片用）
manim -pqh src/viz/animation/kdtree_anim.py QueryScene
manim -pqh src/viz/animation/lsh_anim.py QueryScene
```

**使用真實 LibriSpeech 資料（需安裝 SpeechBrain）：**
```bash
pip install speechbrain torchaudio
python main.py benchmark
```

### 與課程的關聯總結

本專題從四個維度展示了資料結構與演算法的核心概念：

**1. Tree 結構與遞迴**
KD-Tree 是 Binary Search Tree 的多維延伸。每個節點以最高 variance 維度的中位數做切割，遞迴建立二元樹。這與課程中 BST 的插入與分割邏輯直接對應，差異在於切割維度隨深度輪換，讓空間分割更均勻。

**2. Pruning（剪枝）**
KD-Tree 查詢使用 branch-and-bound：若查詢點到切割超平面的距離大於目前第 k 近鄰的距離，整棵子樹可以跳過。這與課程中討論的搜尋樹剪枝（如 alpha-beta pruning）概念相同——不可能有更好答案的分支直接略過。

**3. Hashing 的設計哲學**
LSH 反轉了一般 hash 「避免 collision」的目標：它刻意設計讓相似向量產生相同 hash。Random projection 保留了向量在高維空間的距離關係，使相似的 speaker embedding 落入同一 bucket，實現 O(1) 近似查詢。

**4. Space-Time Tradeoff 的量化**
三種方法的 recall/latency benchmark 直接量化了理論上的取捨：Flat 犧牲速度換精確；LSH 犧牲少量精確換速度；KD-Tree 在高維下兩者皆輸，驗證了 curse of dimensionality 不只是理論，而是可測量的工程現象。

### Demo 影片

> YouTube 連結：（待補）

**影片結構（5 分鐘）：**
- 0:00–1:00　專案介紹：Speaker Verification 問題與三種索引方法
- 1:00–3:00　功能展示：Streamlit UI 查詢、Three.js 3D 點雲、benchmark 結果
- 3:00–5:00　課程關聯：Manim KD-Tree 走訪動畫 + LSH bucket 動畫說明
