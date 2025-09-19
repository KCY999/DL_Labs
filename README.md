# 113-2 TAICA: Deep Learning Labs 

_本專案僅包含本人自行撰寫的程式碼與報告，未公開任何課程講義或教師提供之資源。僅作為個人115推甄備審資料所用，如有疑慮請與我聯絡 (Email: b1126006@cgu.edu.tw)。_

## Lab 2: Binary Semantic Segmentation    
- **要求**:
  - 在 **Oxford-IIIT Pet Dataset** 上進行 **二元語意分割** (前景/背景)。  
  - 自行實作 **U-Net 與 ResNet34+U-Net** 架構，不可直接使用套件內建模型，也不能用 pre-trained weights。  
  - 訓練流程需包含：資料前處理、自訂 loss function (CrossEntropy / BCE)、模型訓練與驗證，並以 **Dice Score** 作為評估指標。  
  - 測試階段需計算平均 Dice Score 並將分割結果可視化。  
- **實作內容**:
  - **模型**：  
    - U-Net：實作 DoubleConv、DownSampling、UpSampling、bottleneck，修改 padding 確保輸出與 mask 尺寸一致。  
    - ResNet34+U-Net：以 ResNet34 為 encoder，搭配 UNet decoder，增加 bridge layer 與 batch normalization。  
  - **訓練流程**：  
    - `train.py`：支援 Early Stopping (patience=7)、AdamW optimizer、最佳權重保存。  
    - `evaluate.py`：計算 validation loss 與 Dice Score。  
    - `inference.py`：測試集推論與 Dice Score 評估。  
  - **資料處理**：  
    - 採用 z-score normalization。  
    - 實作兩種資料增強：`combine` (隨機旋轉/位移/翻轉)、`flip` (水平/垂直翻轉)，並測試不同 `n_aug`。  
- **結果**:  
  - **U-Net**：lr=1e-4、`combine`、`n_aug=1` → **Dice Score = 0.9309**。  

  <img src="Figs\lab2_U-Net_acc.png" alt="U-Net acc" width="400" height="250"/> <img src="Figs\lab2_U-Net_loss.png" alt="U-Net loss" width="400" height="250"/>

  - **ResNet34+U-Net**：lr=1e-3、`combine`、`n_aug=1` → **Dice Score = 0.9478**。  

  <img src="Figs\lab2_Res-U-Net_acc.png" alt="Res-U-Net acc" width="400" height="250"/> <img src="Figs\lab2_Res-U-Net_loss.png" alt="Res-U-Net loss" width="400" height="250"/>

  - **觀察**：  
    - U-Net 對學習率與資料量敏感，需搭配增強與較低學習率才能穩定收斂。  
    - ResNet34+U-Net 受益於 skip connection 與深層結構，在小數據下表現更穩定。  

## Lab 5: Value-Based Reinforcement Learning
- **要求**:
  - **Task 1**：在 CartPole-v1 上實作 Vanilla DQN，使用全連接網路作為 Q-function，並包含經驗回放與目標網路。  
  - **Task 2**：在 Atari Pong-v5 上擴展 DQN，需進行影像前處理 (灰階、縮放、frame stacking)，並使用 CNN 作為 Q-function。  
  - **Task 3**：在 Pong-v5 上整合 Double DQN、Prioritized Experience Replay (PER)、Multi-Step Return，並比較增強版 DQN 與 Vanilla DQN 的學習效率。  
  - 評估方式：需繪製訓練曲線，提交模型快照 (Task1/Task2最佳、Task3在200k–1M steps)，並製作 5–6 分鐘 demo video。  
- **實作內容**:
  - **Task 1**：實作 Vanilla DQN，採用 fully connected NN，replay buffer 用 `deque` + uniform sampling，loss 使用 MSE (Bellman error)。  
  - **Task 2**：改用 CNN (3 conv + 2 fc) 處理影像輸入，調整 lr、epsilon decay、episodes，讓訓練更快收斂。  
  - **Task 3**：  
    - PER：用 TD error 做 priority，支援 add/sample/update，並加上 importance sampling。  
    - Double DQN：分離 action selection (online net) 與 evaluation (target net) 避免 Q-value 高估。  
    - Multi-Step Return：用 `deque` buffer 收集 n 步 transition，減少對 bootstrapped Q 的依賴。  
    - 加入 StepLR 動態 lr 調整、多次 evaluation 取平均、測試 reward shaping 與 epsilon-min 改動。  
- **結果**:
  - **DEMO VIDEO**: https://youtu.be/q2_HySgZReU  

  <img src="Figs\lab5-demo.png" alt="Lab 5 demo" width="500" height="250"/> 

  - **Task 1 (CartPole-v1)**：調整 lr=0.0006、epsilon-decay=0.9998、target-update=300，最終穩定達到 **滿分 500 reward**。  

  <img src="Figs\lab5_task1_eval_reward.png" alt="Lab 5 task1" width="500" height="250"/> 

  - **Task 2 (Pong-v5, Vanilla DQN)**：lr=0.0002、epsilon-decay=0.9997，約 100–150 萬 steps 收斂至 **17–19 分**，僅需 4 小時 (RTX 4060Ti)。  

  <img src="Figs\lab5_task2_eval_reward.png" alt="Lab 5 task2" width="500" height="250"/> 

  - **Task 3 (Pong-v5, Enhanced DQN)**：  
    - 參考 Rainbow baseline 並調整 lr/target update/lr decay。  
    - **v16 配置 (lr=0.0001 + lr scheduler)**：達到 **17–19 分**，兼具收斂速度與穩定性。  

    <img src="Figs\lab5_task3_v16_eval_reward.png" alt="Lab 5 task3 v16" width="500" height="250"/>  

    - **v24 配置 (epsilon-min=0.0)**：在 1M steps 內達成 **平均 reward 19**，符合課程評分標準。   

    <img src="Figs\lab5_task3_v24_eval_reward.png" alt="Lab 5 task3 v24" width="500" height="250"/> 


## Lab 6: Generative Models 
- **要求**:  
  - 實作 **Conditional Denoising Diffusion Probabilistic Model (DDPM)**，根據 **multi-label 條件** (test.json、new_test.json) 產生合成影像。  
  - 使用助教提供的 **ResNet18 evaluator** 計算生成影像的分類正確率 (test.json 與 new_test.json)。  
  - 輸出結果需包含：  
    - 兩組測試資料的影像網格 (test.json、new_test.json)。  
    - 一個 denoising 過程圖 (例：["red sphere", "cyan cylinder", "cyan cube"])。  
  - 撰寫報告，描述模型設計 (架構、embedding、noise schedule、sampling) 與實驗結果。  
- **實作內容**:  
  - 採用 **UNet 架構的 Conditional DDPM**，共測試三種版本：  
    1. **Version 1**：簡化版 UNet，僅在 bottleneck 加入 time embedding，BatchNorm + ReLU。  
    2. **Version 2**：對齊 DDPM 論文，在每層加入 time 與 condition embedding，使用 SiLU，但仍採 BatchNorm。  
    3. **Version 3**：同 Version 2，但將 BatchNorm 換成 **GroupNorm**，改善 embedding 與 normalization 的衝突。  
  - Noise schedule：**linear β** (1e-4 → 0.02, T=1000)。  
  - Loss function：MSE 預測加入的 noise。  
  - 訓練流程：隨機選 timestep t，加噪聲，模型預測噪聲，計算 MSE loss 更新參數。  
- **結果**:  
  - **Version 1**：test=0.71 / new_test=0.82 (500 epochs)，延長至 1000 epochs 無明顯提升。  
  - **Version 2**：test=0.60 / new_test=0.73，BatchNorm 在 embedding 前干擾內部統計，效果下降。  
  - **Version 3**：test=0.86 / new_test=0.87，達到課程最高分標準，生成影像品質穩定，denoising 過程清晰一致。  
    - denosing process of **Version 3**   

    <img src="Figs\lab6_denoise_process.png" alt="Lab 6 v3 denoising process" width="1000" height="100"/> 

    - images grid of test of **Version 3**  

    <img src="Figs\lab6_test_img_grid.png" alt="Lab 6 v3 test_img_grid" width="500" height="250"/>  

    - images grid of _new_test of **Version 3**  

    <img src="Figs\lab6_new_test_img_grid.png" alt="Lab 6 v3 new_test_img_grid" width="500" height="250"/>  


  - **關鍵發現**：  
    - **embedding 要在每層注入 (而非只在 bottleneck)**。  
    - **GroupNorm > BatchNorm**，更適合 diffusion 中帶有 condition embedding 的架構。  
