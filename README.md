## [How High are We? Large-Scale Building Height Estimation Using Sentinel-1 Sar and Sentinel-2 Msi Time Series] (https://www.sciencedirect.com/science/article/pii/S0034425724005820)

### Building Footprint Segmentation & Height Estimation

### Abstract: 
Accurate building height estimation is essential to support urbanization monitoring, environmental impact analysis and sustainable urban planning. However, conducting large-scale building height estimation is a challenging task. While Deep Learning (DL) has proven effective for large-scale mapping, the lack of advanced DL models specifically tailored for height estimation remains a challenge, particularly when using open source Earth Observation data. In this study, we propose an advanced DL model (T-SwinUNet) for large-scale building height estimation leveraging Sentinel-1 Synthetic Aperture Radar and Sentinel-2 MultiSpectral Instrument time series.In the proposed T-SwinUNet, the semantic feature learning capabilities of the efficientnet encoder are combined with the local/global feature comprehension capabilities of Swin transformers. A temporal attention module is added to learn the correlation between constant and variable features of building objects over time which not only helps in differentiating building objects from the surroundings but also in learning salient features for building height estimation. The model is trained on a multi-task to predict both building height and footprint at 10 m spatial resolution. The model is evaluated on data from the Netherlands, Switzerland, Estonia, and Germany. The extensive evaluation and comparison with state-of-the-art DL models show that our proposed T-SwinUNet model yields Root Mean Square Error (RMSE) of 1.89 m, surpassing the state-of-the-art at 10m spatial resolution. Further assessment at 100 m resolution shows that our predicted building heights (0.29 m RMSE, 0.75 R^2) also outperformed the global building height product GHSL-Built-H R2023A product(0.56 m RMSE and 0.37 R^2).

#### Manuscript 
(OLD: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4762421)

#### Also at EGU2024 : 
(https://meetingorganizer.copernicus.org/EGU24/EGU24-4493.html)
