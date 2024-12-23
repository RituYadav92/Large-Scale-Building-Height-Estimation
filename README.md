## [How High are We? Large-Scale Building Height Estimation Using Sentinel-1 Sar and Sentinel-2 Msi Time Series](https://www.sciencedirect.com/science/article/pii/S0034425724005820) [Remote Sensing of Environment] 

We propose T-SwinUNet, an advanced DL model for large-scale building height estimation leveraging Sentinel-1 SAR and Sentinel-2 multispectral time series. The model was trained and evaluated on data from the Netherlands, Switzerland, Estonia, and Germany, and its generalizability is evaluated on an out-of-distribution (OOD) test set from ten additional cities from other European countries. T-SwinUNet predicts building height with a Root Mean Square Error (RMSE) of 1.89 m, outperforming state-of-the-art models at 10 m spatial resolution. Its strong generalization to the OOD test set (RMSE of 3.2 m) underscores its potential for low-cost building height estimation across Europe, with future scalability to other regions. Furthermore, the assessment at 100 m resolution reveals that T-SwinUNet (0.29 m RMSE, 0.75 R^2) also outperformed the global building height product GHSL-Built-H R2023A product(0.56 m RMSE and 0.37 R^2). 

<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/dataset_location.png" alt="Sites" width="500" height="400">

### 🎉 Manuscript
https://www.sciencedirect.com/science/article/pii/S0034425724005820

Also at  👉 [EGU 2024](https://meetingorganizer.copernicus.org/EGU24/EGU24-4493.html) & 
         👉 [ESA URBIS 2024](https://www.conftool.pro/urbis24/index.php?page=browseSessions&form_session=71&presentations=hide)


### 🛠️ Setup
create the conda environment via

```bash
conda env create -f environment.yml
```

### 🏋️‍♂️ Training
Run the python script `train.py` as follows

```bash
python train.py \
    --exp_root 'CKPT PATH' \
    --config_file './configs/tswin_unet/exp3.yaml' \
    --train-df "TRAIN DATA LIST CSV" \
    --data_root "TRAIN DATA PATH"
```
###  🚀 Inference
Run the python script `inference.py` as follows
```bash
python predict.py \
    --config_file './configs/tswin_unet/exp3.yaml' \
    --output_root 'PREDICTION OUTPUT PATH' \
    --exp_root 'CKPT PATH' \
    --test-df "TEST DATA LIST CSV" \
    --data_root "TEST DATA PATH"
```

### 📈 Results

<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/Quant.jpg" alt="Sites" width="900" height="145">
<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/COR.jpg" alt="Sites" width="680" height="350">
<img src="https://github.com/RituYadav92/Large-Scale-Building-Height-Estimation/blob/main/TSwinUnet/assets/figures/GEE_vis.jpg" alt="Sites" width="900" height="450">

## 🎓 Citation

Please cite our paper:

```bibtex
@article{yadav2025high,
  title={How high are we? Large-scale building height estimation at 10 m using Sentinel-1 SAR and Sentinel-2 MSI time series},
  author={Yadav, Ritu and Nascetti, Andrea and Ban, Yifang},
  journal={Remote Sensing of Environment},
  volume={318},
  pages={114556},
  year={2025},
  publisher={Elsevier}
}
```

### 👋 Contact Info.:
Ritu Yadav (email: er.ritu92@gmail.com)
