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
| Week 7 | Proposal 繳交、建立環境、載入 VoxCeleb embedding |
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
<!-- 完成了什麼 -->

### 遇到的困難
<!-- 遇到什麼問題、如何解決或打算如何解決 -->

### 下一步計畫
<!-- 接下來要做什麼 -->

### 與課程的關聯
<!-- 到目前為止，你的實作中哪些部分與課程內容有關？關係是什麼？ -->

---

## Final Report

### 專案說明
<!-- 完整描述你的專案做了什麼 -->

### 使用方式
<!-- 如何編譯、執行、使用你的程式 -->

### 與課程的關聯總結
<!-- 總結你的專題與進階程式設計及資料結構課程之間的關聯 -->
