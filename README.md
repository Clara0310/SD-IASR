
# SD-IASR: 基於譜解耦與意圖感知的序列推薦系統

本專案為論文 **「基於譜理論的意圖感知序列推薦系統 (Spectral Disentangling Intent-Aware Sequential Recommendation, SD-IASR)」** 的官方 PyTorch 實作。

## 1. 核心模型架構
SD-IASR 旨在解決傳統推薦系統中商品關係耦合的問題。透過譜圖理論 (Spectral Graph Theory)，我們將商品關係解耦為「相似性」與「互補性」，並結合雙通道 Transformer 捕捉使用者的動態意圖。

* **商品關聯圖建構模組**：整合使用者行為記錄，將商品關係建模為兩張語義獨立的圖（相似圖與互補圖）。
* **譜關係解耦模組**：利用圖拉普拉斯矩陣設計低通與中通濾波器，分別捕捉相似性 (Similarity) 與互補性 (Complementarity) 結構。
* **序列編碼與意圖捕捉模組**：採用雙通道 Transformer 架構與 **Fusion Pooling** 策略，融合近期與全局意圖表徵。
* **使用者意圖預測模組**：使用點積與雙線性轉換計算分數，並透過動態權重 $\alpha$ 進行自適應融合。

---

## 2. 運行環境
本專案開發環境基於 Python 3.9。
* **Python**: 3.9
* **PyTorch**: 1.13.0
* **Torchvision**: 0.14.0
* **Torchaudio**: 0.13.0
* **Transformers**: 用於 BERT 語義嵌入生成

安裝依賴套件：
```bash
pip install -r requirements.txt

```

---

## 3. 資料前處理流水線

請依照下列順序執行 `data_preprocess/` 目錄下的腳本：

1. **特徵過濾**：
`python 1_feature_filter.py $dataset` (過濾類別或價格資訊不全的商品)
2. **邊提取**：
`python 2_edge_extractor.py $dataset` (分別產出 sim.edges 與 cor.edges)
3. **圖建構與格式化**：
`python 3_edge_filter.py $dataset` 與 `python 4_data_formulator.py $dataset`
4. **類別嵌入生成**：
`python 5_embs_generator.py $dataset` (使用 BERT 產生類別嵌入)
5. **行為序列建構 (核心創新)**：
`python 6_seq_constructor.py $dataset` (建立使用者動態行為序列 )
6. **資料集分割**：
`python 7_dataset_split.py $dataset` (產出最終訓練用的 .npz 檔案)

---

## 4. 訓練與評估


我們建議使用根目錄下的 `run.sh` 腳本啟動模型訓練，這會自動處理參數傳遞並備份實驗設定：

```bash
# 賦予執行權限並啟動
chmod +x run.sh
./run.sh

### 關鍵超參數：

* `--low_k` / `--mid_k`：控制譜濾波器的頻率集中度
* `--alpha` / `--beta` / `--lambda_3`：調節聯合損失函數中各項的重要性

---

## 5. 誌謝

本程式碼結構參考了 **SR-Rec** 專案的實作基礎。

```

```