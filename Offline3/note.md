# 作業概述

Offline 3 分為兩部分：

1. 實作 diffusion model
   - 實作後使用該模型產生圖片。
2. 使用 Hugging face 上現有的模型，設計不需訓練的 diffusion
   - 需要產生一組光學錯覺的圖片，上下顛倒會產生不同的影像主題。

作業不得使用額外的訓練資料，預訓練的權重等等。

# 作業細節

這份作業的難度非常高，且 diffusion model 需要大量時間。以至於這份作業最後有被延期，且在二手拍上有人徵求家教。助教又為交大創造了工作機會。

也有人特別為此買了 Kaggle 的服務以訓練更好的模型。

作業都公布在 Kaggle 上。

## Diffusion

diffusion model 的部分， U-Net 的內容都由 sample code 提供，實際要處理的只有 diffusion 的數學公式，在 SPEC 中有提供。我聽聞有些人嘗試用城市中部份已經定義好的變數，不過實際上未在 sample code 中使用或提及的變數都不該被使用。

diffusion 分 3 部分：初始化、前傳播、和後傳播。除了實作公式以外，最好注意變數的型別並考慮張量的大小，部分程式不處理張量也不會出錯，但最好都一起處理。

diffusion 實作後， sample code 在運作時會讓模型生成圖片，因此還需要紀錄每個 epoch 的訓練效果。最後的結果需要檢驗 FID 和 AFD 的效果。如果不修改 sample code ， diffusion 預設的參數和 scheduler 只會產生接近 default baseline 的結果，不會有好的分數。

## Optical Illusion

光學錯覺要準備兩個 prompt ，代表上下兩個方向圖片主題。以此調用 Hugging face 的模型。主要部份是實作 denoising 的部分， sample code 有提供單一圖片的版本，因此較前一部份簡單，需要的時間也較少。

# Demo

此作業不須 demo ，但是 diffusion 的圖片結果需要分享在 google drive 。