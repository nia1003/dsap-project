# ANN-Bench: Approximate Nearest Neighbor Search for Speech Embeddings

## Proposal Report

### 動機與目標

Speech Language Models（語音語言模型）將音訊資料編碼為高維向量（embedding），廣泛應用於說話者辨識、語音檢索、語意語音搜尋等任務。然而，當 embedding 資料庫規模擴大時，如何在高維空間中快速找到最相似的向量，成為關鍵的基礎設施問題。

現有工具（如 FAISS）將這個問題抽象化，使用者往往不理解其底層運作。本專題的目標是**從頭實作三種向量索引方法**，並在 speech embedding 資料集上進行嚴謹的效能比較，從演算法層面理解 Approximate Nearest Neighbor（ANN）搜尋的核心原理與取捨。

### 預期功能

1. **三種索引方法實作**
   - Flat Search（暴力搜尋，作為 baseline）
   - KD-Tree（空間分割樹，適合低維度精確搜尋）
   - LSH（Locality-Sensitive Hashing，適合高維度近似搜尋）

2. **效能 Benchmark**
   - 在 speech embedding 資料集（如 LibriSpeech speaker embeddings）上比較各方法的 recall@k 與 query latency
   - 分析維度、資料量對各索引的影響

3. **互動式查詢介面**
   - CLI 或簡易 Web UI，輸入一段音訊或 embedding，返回最相似的 Top-K 結果

4. **視覺化**
   - 將高維 embedding 降維（PCA/t-SNE）後，動畫呈現 KD-Tree 的空間分割與查詢走訪過程，以及 LSH 的 bucket 分群效果

### 使用技術

- **語言**：Python
- **核心實作**：純 Python（不依賴 FAISS 等現成 ANN 函式庫）
- **資料集**：LibriSpeech / VoxCeleb speaker embeddings（或使用預訓練模型提取）
- **視覺化**：matplotlib、plotly
- **UI**：CLI（argparse）或 Streamlit

### 時程規劃

| 週次 | 預計進度 |
|------|---------|
| Week 7 | Proposal 繳交、確認資料集與環境 |
| Week 8 | 實作 Flat Search baseline + embedding 資料載入 |
| Week 9 | 實作 KD-Tree 索引（建樹 + 查詢） |
| Week 10 | 實作 LSH 索引（hash function 設計 + bucket 查詢） |
| Week 11 | Prototype 繳交、完成初步 benchmark 比較 |
| Week 12 | 設計 CLI / Web UI 查詢介面 |
| Week 13 | 實作視覺化（KD-Tree 走訪動畫、LSH bucket 分群） |
| Week 14 | 完整 benchmark、撰寫分析報告 |
| Week 15 | Final Report 繳交、錄製 demo 影片 |

### 與課程的關聯

本專題直接對應課程中的多個核心主題：

- **Tree**：KD-Tree 是 Binary Search Tree 的高維延伸，建樹過程涉及遞迴分割與中位數選取，查詢涉及剪枝（pruning）策略
- **Hashing**：LSH 的核心是設計對相似向量產生相同 hash 的函數族，涉及 hash collision 的設計與控制
- **Space & Time Tradeoff**：三種方法體現了精確度、建立時間、查詢速度之間的根本取捨，是演算法設計的核心議題
- **Dimensionality & Complexity**：分析各方法在不同維度下的時間複雜度（KD-Tree 在高維退化為 O(n)，LSH 保持次線性）

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
