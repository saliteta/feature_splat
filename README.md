# Feature Splat
- This is a backend implementation of SemanticField
- We modify the cuda kernel so that we can tuning up to 1024 dimension features
- This repository is most inherent from gsplat, we only modify the rasterization cuda kernel forward and backward 

## Detailed Implementation
- We imeplement: feature_splat/cuda/csrc/rasterization.cu 
- We by pass most of the back propagation path to speed up and reduce the memory consumption
- By pass means, we train the original Gaussian Splatting (means, quaternion, alpha, scales, RGB) before hand
- In the feature splat,we stablize all parameters except features only. Therefore the gradient calculation do not need to store features
- Current memory consumption is stablize at around 31GB

## Optimization
![Profile](assets/profiling.png)
- Our code has optimized for calculation. it is calculation driven instead of memory driven as one can observe from above profile image
- Time for training: Training 30000 iteration on 755*1006*(up to 1024) image will take 5 hours at most
- The feature size will always round up to next binary power. Eg: (768 -> 1024) (244 -> 256), when feature dimension decrease by half, the time will shrink by half

## Install
- First install nerfstudio as follow: https://docs.nerf.studio/
- Second pip install .

## Usage
To use it please follow the front end application usage example, like semantic field