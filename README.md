# Task1 - VCL팀

2024.09.12 작성
- 본 대회의 4가지 Task 중에서 패션 이미지의 속성을 분류하는 Task 1에 대한 접근법을 설명
- Task 1은 하나의 패션 이미지에 대하여 '일상성', '성별', '장식성' 3가지 속성 각각의 클래스를 분류해야함
- 앙상블기법 사용 불가 및 모델 용량 25MB 이하의 제한 조건이 존재

1. 전체적인 모델 구조 및 학습 방법
2. 모델 학습 환경
3. 실험환경구성
4. 필요한 패키지 설치 
5. 모델 학습
___

### 1. 전체적인 모델 구조 및 학습 방법
![p1](/src/ppt_p1.png)
![p2](/src/ppt_p2.png)
![p3](/src/ppt_p3.png)

### 2. 모델 학습 환경

- Ubuntu 20.04
- Python 3.8.0
- [PyTorch 1.12.1](https://pytorch.org/get-started/previous-versions/)
- [CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)
- NVIDIA RTX 3090 GPU 1개


### 3. 실험환경구성

아래의 커맨드를 통해 가상환경을 생성하고 [PyTorch(1.12.1)](https://pytorch.org/get-started/previous-versions/) 를 설치합니다.

```bash
conda env create -f environments.yaml
conda activate task1
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

추가적으로 사용한 라이브러리들을 설치합니다.

```bash
pip install transformers==4.27.4 timm==1.0.9 scikit-image==0.20.0 scikit-learn==1.2.2 scipy==1.9.1
pip install pandas tqdm
```

### 4. 추가적으로 사용한 코드

- [utils/shceduler.py](https://gaussian37.github.io/dl-pytorch-lr_scheduler/) : cosine scheduler 를 사용하기 위함. (train.py)
- [utils/losses.py](https://github.com/snap-research/EfficientFormer/blob/main/util/losses.py) : Knowledge Distillation Loss를 사용하기 위함. (train.py)
- [clip_model.py](https://github.com/patrickjohncyh/fashion-clip) : FashionCLIP 모델을 사용하기 위함. (train.py)
- [tiny_vit.py](https://github.com/wkcn/TinyViT/blob/main/models/tiny_vit.py) : [`TinyViT`](https://github.com/wkcn/TinyViT/tree/main?tab=readme-ov-file) TinyVIT 모델을 사용하기 위함. (networks.py)

---
### 5. 모델 학습

- 데이터셋 준비
    - 대회에서 제공 받은 `Dataset` 폴더의 심볼릭 링크를 생성하여 사용하였습니다.
    
    ```bash
    ln -s {Dataset path} ./Dataset
    ````

#### 5-1. Teacher 모델 학습
- 학습
    - train.sh 파일을 사용하여 Teacher 모델 학습을 시작합니다.

        ```bash
        python train.py \
            --version teacher \
            --seed 214 \
            --lr 1e-6 \
            --optimizer Adam \
            --epochs 10 \
            --batch-size 64 \
            --weight-decay 0.2 \
            --loss-weight True \
        ```
        --version : log, model 파일을 저장할 폴더이름 입니다.  
        학습을 시작하면 `teacher/` 폴더가 logs, model 폴더안에 생성되며 학습이 시작됩니다.  

    - logs/teacher 폴더 안에 `teacher_training.log` 로그 파일이 생성되어 학습 로그를 기록합니다.

    - model/teacher 폴더 안에 모델의 가중치가 1epoch마다 저장됩니다.

#### 5-2. Student 모델 학습
- 학습
    - train.sh 파일을 사용하여 Teacher 모델 학습을 시작합니다.

        ```bash
        python train.py \
            --version student \
            --seed 214 \
            --lr 1e-3 \
            --min-lr 4e-5 \
            --cos-gamma 0.9 \
            --warmup-epoch 5 \
            --decay-epoch 30 \
            --optimizer AdamW \
            --epochs 10 \
            --batch-size 128 \
            --weight-decay 0.01 \
            --loss CE \
            --ls 0.1 \
            --kd-learning True \
            --teacher teacher/model_5.pt \
            --distillation-type soft \
            --distillation-tau 4 \
            --distillation-alpha 0.9 \
            --loss-weight True \
            --scheduler True \
        ```
    - Teacher 모델 학습과 동일하게 가중치 파일(.pt) 과 log파일(.log) 이 저장됩니다.  