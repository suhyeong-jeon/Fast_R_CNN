# Fast_R_CNN   

##### Fast-R-CNN은 R-CNN의 문제점을 보완했다. R-CNN은 2000장의 region proposals을 CNN 모델에 입력시켜 각 모델에 대해 독립적으로 학습을 시켜 많은 시간이 필요했지만, Fast-R-CNN은 단 1장의 이미지를 입력받고, region proposals의 크기를 warp시킬 필요 없이 RoI(Region of Interest) pooling을 사용해 고정된 크기의 feature vector를 fully connected layer에 전달하고, multi-task-loss를 사용해 모델을 한번에 학습시킨다. 이를 통해 학습 및 detection 시간이 크게 감소했다.   

<hr>

### Main Ideas   
#####   

### 1. RoI(Region of Interest) Pooling   
#####   
##### RoI는 feature map에서 region proposals에 해당하는 관심 영역(Region of Interest)을 지정한 크기의 grid로 나눈 후 max pooling을 수행하는 방법이다. 각 channel별로 독립적으로 수행하며, 이 방법으로 고정된 크기의 featuremap을 출력하는 것이 가능하다.   
#####   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/d4175d5f-03c5-4320-8476-ebdad5dce643"></p>

#####   
##### 1. 원본 이미지를 CNN 모델에 통과시켜 feature map을 얻는다.   
##### - 800 * 800 크기의 이미지를 VGG 모델에 입력해 8 * 8 크기의 feature map을 얻는다.   
##### - sub-sampling ratio는 1/100이라고 할 수 있다.   
##### 2. 동시에 원본 이미지에 대하여 Selective Search 알고리즘을 적용하여 region proposals을 얻는다.   
##### - 원본 이미지에 Selective Search 알고리즘을 적용해 500 * 700 크기의 region proposal을 얻는다.   
##### 3. Feature map에서 각 region proposals에 해당하는 영역을 추출한다. 이 과정은 Region Projection을 통해 가능하다. Selective Search를 통해 얻은 Region Proposals은 sub-sampling을 거치지 않은 반면, 원본 이미지의 feature map은 sub-sampling 과정을 여러번 거쳐 크기가 작아졌기 때문에, 작아진 feature map에서 region proposals이 encode하고 있는 부분을 찾기 위해 작아진 feature map에 맞게 region proposals을 투영해주는 과정이 필요하다. 이 과정은 region proposals의 크기와 중심 좌표를 sub sampling ration에 맞게 변경시켜줌으로써 가능하다.   
##### - Region Proposal의 중심점 좌표, width, height와 sub-sampling ratio를 활용해 feature map으로 투영한다.   
##### - Feature map에서 region proposal에 해당하는 5 * 7 영역을 추출한다.   
##### 4. 추출한 RoI feature map을 지정한 sub-window의 크기에 맞게 grid로 나눠준다.   
##### - 추출한 5 * 7 크기의 영역을 지정한 2 * 2 크기에 맞게 grid를 나눠준다.   
##### 5. grid의 각 셀에 대하여 max pooling을 수행해 고정된 크기의 feature map을 얻는다.   
##### - 각 grid 셀마다 max pooling을 수해앟여 2 * 2 크기의 feature map을 얻는다.   
#####   
##### 이처럼 미리 지정한 크기의 sub-window에서 max pooling을 수행하다보니 region proposal의 크기가 서로 달라도 고정된 크기의 feature map을 얻을 수 있다.   
#####   

### 2. Multi-Task Loss   
#####   
##### Fast-R-CNN 모델에서는 feature vector를 multi-task-loss를 사용해 Classifier와 Bounding Box Regressor를 동시에 학습시킨다. 두 모델을 한번에 학습시키기 때문에 R-CNN 모델과 같이 각 모델을 독립적으로 학습시켜야 하는 번거로움이 없다는 장점이 있다.   
#####   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/f060cbad-4d41-4400-b3b3-3c55376aa027"></p>

#####   

### 3. Hierarchical Sampling   
#####   
##### R-CNN 모델은 학습 시 region proposal이 서로 다른 이미지에서 추출되고, 이로 인해 학습 시 연산을 공유할 수 없다는 단점이 있다. 논문의 저자는 학습 시 feature sharing을 가능하게 하는 Hierarchical sampling 방법을 제시한다.   
##### SGD mini-batch를 구성할 때 N개의 이미지를 sampling하고, 총 R개의 region proposal을 사용한다고 할 때, 각 이미지로부터 R/N개의 region proposals를 sampling하는 방법이다. 이를 통해 같은 이미지에서 추출된 region proposals끼리는 forward, backward propogation 시, 연산과 메모리를 공유할 수 있다.   
#####   

### 4. Truncated SVD   
#####   
##### Fast-R-CNN은 detection시 RoI를 처리할 때 fc layer에서 많은 시간이 걸리는데 detection 시간을 감소시키기 위해 Truncated SVD(Singular Vector Decomposition)을 통해 fc layer를 압축하는 방법을 제시한다.   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/4b94d8b9-2f0c-4bd1-b33a-bf0df6522313"></p>

##### 행렬 A를, m * m크기인 U, m * n 크기인 Σ, n * n크기인 V**T로 특이값 분해(SVD)하는 것을 Full SVD라고 한다.   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/40e5b868-5776-41ad-84eb-19dbd6d08369"></p>

##### Truncated SVD는 Σ의 비대각 부분과 대각 원소 중 특이값이 0인 부분을 모두 제거하고, 제거된 Σ에 대응되는 Y, V 원소도 함께 제거하여 차원을 줄인 형태다. Ut의 크기는 m * t고, Σt의 크기는 t * t, Vt는 t * n이다. 이렇게 되면 행렬 A를 상당히 근사하는것이 가능하다.   
##### Truncated SVD를 통해 Detection 시간이 30% 정도 감소되었다고 한다.   

<hr>

### Training Fast R-CNN   
#####   
##### Fast R-CNN 모델의 학습 과정을 보도록 하자. 하나의 이미지가 입력되었을 때를 가정하고 전체 학습과정을 살펴볼 것이다.   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/12ff272d-fafa-43ed-bd3d-200ae75af038"></p>

##### 1. Initializing pre-trained network   
#####   
##### Feature map을 추출하기 위해 VGG16 모델을 사용한다. 먼저 VGG16을 detection task에 맞게 변형시켜줘야한다.   
##### 1) VGG16 모델의 마지막 max pooling layer를 RoI pooling layer로 대체한다. 이 때 RoI pooling을 통해 출력되는 feature map의 크기인 H, W는 후속 fc layer와 호환 가능한 크기인 7 * 7로 설정해준다.   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/5896ff0c-cdb1-47cc-8155-bd71387f9b31"></p>

##### 2) 네트워크의 마지막 fc layer를 2개의 fc layer로 대체한다. 첫 번째 fc layer는 K개의 class와 배경을 포함한 K+1개의 output unit을 가지는 Classifier고, 두 번째 fc layer는 각 class 별로 bounding box의 좌표를 조정하여 (K+1)*4개의 output unit을 가지는 bounding box regressor다.   
##### 3) conv lyaer3까지의 가중치값는 freeze(고정)시켜주고, 이후 conv layer4부터 fc layer3까지의 가중치값이 학습될 수 있도록 fine tuning 해준다.   
##### 4) 네트워크가 원본 이미지와 Selective Search 알고리즘을 통해 추출된 Region Proposals 집합을 입력으로 받을 수 일도록 변환시켜 준다.   
#####   

##### 2. Region Proposal by Selective Search   
#####   
##### 원본 이미지에 대해 Selective Search 알고리즘을 적용해 미리 region proposals를 추출한다.   
##### Input : image, Process : Selective Search, Output : 2000 region proposals   
#####   

##### 3. Feature Extraction(~layer13 pre-pooling) by VGG 16   
#####   
##### VGG16모델에 224*224*3 크기의 원본 이미지를 입력하고, layer13까지의 feature map을 추출한다.   
##### Input : 224*224*3 sized image, Process : Feature extraction by VGG16, Output : 14*14*12 feature maps   
#####   

##### 4. Max pooling by RoI Pooling   
#####   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/f4966c51-70c0-4e28-bbfe-9e507b1612fe"></p>

##### Region proposals를 layer13을 통해 출력된 feature mpa에 대하여 RoI projection을 진행한 후, RoI pooling을 수행한다. 이를 통해 고정된 7 * 7 크기의 feature map을 추출한다.   
##### Input : 14 * 14 sized 512 feature maps, 2000 regoin proposals, Process : RoI Pooling, Output : 7*7*512 maps   
#####   

##### 5. Feature Vector Extraction by FC Layers   
#####   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/920434f8-899a-4c81-92a0-5e897ee1a167"></p>

##### Region proposals별로 7*7*512(=25088)의 feature map을 flatten한 후 fc layer에 입력하여 fc layer를 통해 4096 크기의 feature vector를 얻는다.   
##### Input : 7*7*512 sized feature map, Process : Feature extraction by fc layers, Output : 4096 sized feature vector   
#####   

##### 6. Class Prediction by Classifier   
#####   
##### 4096 크기의 feature vector를 K개의 class와 배경을 포함해 K+1개의 ouptut unit을 가진 fc layer에 입력한다. 하나의 이미지에서 하나의 region proposal에 대한 class prediction을 출력한다.   
##### Input : 4096 sized feature vector, Process : class prediction by Classifier, Output : K+1 sized vector(class score)   
#####   

##### 7. Detailed Localization by Bounding Box Regressor   
#####   
##### 4096 크기의 feature vector를 class별로 bounding box의 좌표를 예측하도록 (K+1) * 4개의 output unit을 가진 fc layer에 입력한다. 하나의 이미지에서 하나의 region proposal에 대한 class별로 조정된 bounding box 좌표값을 출력한다.   
##### Input : 4096 sized feature vector, Process : Detailed localization by Bounding box regressor, Output : (K+1)*4 sized vector   
#####   

##### 8. Train Classifier and Bounding Box Regressor by Multi-Task Loss   
#####   
##### Multi-Task-Loss를 사용해 하나의 region proposal에 대한 Classifier와 Bounding box regrssor의 loss를 반환한다. 이후 Back propogation을 통해 두 모델을 한번에 학습시킨다.   
##### Input : (K+1) sized vector(class score), (K+1)*4 sized vector, Process : Calculate Loss by Multi-Task Loss function, Output : Loss   

<hr>

### Detection Fast R-CNN   

<p align="center"><img src="https://github.com/suhyeong-jeon/Fast_R_CNN/assets/70623959/c8150464-2cdd-40b9-b3ef-fc8784b66f8b"></p>

##### Fast R-CNN의 detection시 동작을 살펴보도록 하자. Detection시 동작 순서는 학습 과정과 크게 다르지 않는데, 4096 크기의 feature vector를 출력하는 fc layer에 Truncated SVD를 적용한다는 점에서 차이가 있다. 또한 예측한 bounding box에 대하여 Non Maximum Supression 알고리즘이 추가되어 최적의 Bounding Box만을 출력하게 된다.

<hr>

##### Fast R-CNN은 R-CNN보다 학습속도가 9배 이상 빠르고 detection 시간이 줄어들었다. 성능역시 향상되었으며 multi-task-loss를 사용해 single stage로 여러 모델을 학습시킬 수 있다는 장점과, 학습으로 인해 네트워크의 가중치 값을 back propagation을 통해 update할 수 있다는 장점이 있다.

<hr>

### References   
##### [Fast R-CNN Thesis](https://arxiv.org/pdf/1504.08083.pdf)   
##### [Truncated SVD](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-20-%ED%8A%B9%EC%9D%B4%EA%B0%92-%EB%B6%84%ED%95%B4Singular-Value-Decomposition)   
##### [Fast R-CNN](https://herbwood.tistory.com/8)   
##### [PyTorch Fast R-CNN](https://herbwood.tistory.com/9?category=867198)   
##### [Fast R-CNN Code](https://github.com/gary1346aa/Fast-RCNN-Object-Detection-Pytorch/blob/master/README.ipynb)


