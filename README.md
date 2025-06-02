# Kaggle - Predict Calorie Expenditure | 特徵工程與 XGBoost 建模流程

本筆記記錄我參加 [Kaggle Playground Series - S5E5](https://www.kaggle.com/competitions/playground-series-s5e5) 的完整建模流程，目標是根據運動時長、心率、體溫等資訊，**預測個體運動所消耗的卡路里（Calories）**。

本次競賽成績雖然僅獲得前 40.76% 成績，但透過這次實作，我深入學會了從 EDA、分箱、交叉特徵到模型調參的完整資料建模流程。
後續會持續在資料科學領域精進，挑戰更多比賽！

最終成績（Leaderboard）
	•	Public Score: 0.05971
	•	最終排名：1761 / 4318
	•	Percentile：Top 40.76%
    
---

## 專案流程總覽

1. **資料載入與初步處理**
2. **特徵工程**
   - 分箱與類別交叉特徵（AgeGroup, Duration_group 等）
   - 生理指標推估（MET, HR_max, BMR, TDEE）
   - 數值交叉特徵與能量預估
   - 分群特徵（KMeans_cluster）
3. **偏態處理與離群值清理**
   - 針對右偏與左偏數值欄位進行 `log1p` / `Yeo-Johnson` 轉換
   - ~~移除離群值（使用 IQR 原則）~~
4. **類別特徵 One-Hot 編碼**
   - 類別欄位轉為虛擬變數並與測試集對齊
5. **XGBoost 模型訓練與 K-Fold 驗證**
   - 使用最佳參數訓練模型（由 Optuna 自動調參產生）
   - 執行 K-Fold 交叉驗證評估 RMSLE 效果
6. **預測與產出 submission.csv**
   - 輸出預測結果與模型儲存（.pkl）

---

## 使用的最佳模型參數（由自動調參獲得）

由 Optuna 搜尋獲得以下 XGBoost 超參數，效果優於其他 ensemble 模型與自定義設定：

```python
best_params = {
    'learning_rate': 0.02071,
    'max_depth': 10,
    'n_estimators': 2884,
    'colsample_bytree': 0.5193,
    'gamma': 6.8068,
    'min_child_weight': 1,
    'reg_lambda': 8.4324,
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'mae',
    'random_state': 42
}

---

執行環境
	•	Python 3.8+
	•	Jupyter Notebook
	•	XGBoost
	•	Scikit-learn
	•	Pandas / NumPy
	•	Joblib / TQDM