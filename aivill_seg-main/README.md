# aivill_seg
# 2022 인공지능 온라인 경진대회 깃허브
## 준지도학습 기반의 항만 구조물 객체 분할 문제 에이아이빌 팀

## 모델 구축에 필요한 모듈 및 버전
- python 3.8.5

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# 필요 학습 내용
- 준지도학습(Semi-supervised Learning)

- 객체 분할(Image semantic segmentation)

- mIoU(mean Intersection over Union) 계산

![http://url/to/img.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F2m2Ai%2FbtqCynS9syS%2FihBYPisF7xWXgrt8cSk87K%2Fimg.png)

IoU = 교집합 영역 넓이 / 합집합 영역 넓이

mIoU정의: 각 사진들의 IoU값의 전체 평균.

IoU: Segmentation데이터에서 Ground Truth와 Prediction 사진의 영역의 넓이 교집합 / 합집합

# 유용한 페이지
- 이미지 세그멘테이션 한 번에 끝내기
https://youtu.be/sbcC6N5xfy8
3시간 반 가량의 이미지 세그멘테이션의 이해를 도울 수 있는 강의

- 파이토치 한 번에 끝내기
https://youtu.be/k60oT_8lyFw
파이토치 간단 강의

- segmentaion 벤치마크
https://paperswithcode.com/task/semantic-segmentation

- SSL segmentation 벤치마크
https://paperswithcode.com/task/semi-supervised-semantic-segmentation



# 깃허브
- segmentation-models-pytorch
https://github.com/qubvel/segmentation_models.pytorch
segmentation pytorch 모델 생성 모듈

- Reco 모델
https://github.com/a22106/reco
semi-supervised segmentation에서 좋은 성능을 낸 reco 모델

- albumentations
https://github.com/albumentations-team/albumentations
이미지 증강 모듈

- polygon pp

https://github.com/fidler-lab/polyrnn-pp-pytorch
ploygon annotation 이미지 segmentation

[참고 문헌](https://openaccess.thecvf.com/content_cvpr_2018/html/Acuna_Efficient_Interactive_Annotation_CVPR_2018_paper.html)


# 관련 논문
- Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels(SOTA)
[https://arxiv.org/abs/2203.03884](https://arxiv.org/abs/2203.03884)

- Bootstrapping Semantic Segmentation with Regional Contrast
https://arxiv.org/abs/2104.04465: 

- Semi-Supervised Semantic Segmentation with Cross-Consistency Training
https://paperswithcode.com/paper/semi-supervised-semantic-segmentation-with-1


# [논문 리뷰](https://github.com/a22106/AI_Paper_Review)
## Segmentation

- [DeepLabV1(2014)](https://deep-learning-study.tistory.com/564)

- [FCN(2015)](https://deep-learning-study.tistory.com/562)

- [DeConvNet(2015)](https://deep-learning-study.tistory.com/565)

- [DilatedNet(2015)](https://deep-learning-study.tistory.com/664), paper [[pdf](https://arxiv.org/abs/1511.07122)]

- PyTorch 구현 코드로 살펴보는 [SegNet(2015)](https://deep-learning-study.tistory.com/672), paper [[pdf](https://arxiv.org/pdf/1511.00561.pdf)]

- [PSPNet(2016)](https://deep-learning-study.tistory.com/864), paper [[pdf](https://arxiv.org/abs/1612.01105)]

- [DeepLabv3(2017)](https://deep-learning-study.tistory.com/877), paper [[pdf](https://arxiv.org/abs/1706.05587)]

- [Mask R-CNN(2017)](https://deep-learning-study.tistory.com/571)

- [PANet(2018)](https://deep-learning-study.tistory.com/637), paper [[pdf](https://arxiv.org/abs/1803.01534)]

- [Panoptic Segmentation(2018)](https://deep-learning-study.tistory.com/861), paper [[pdf](https://arxiv.org/abs/1801.00868)]

- Weakly- and Semi-Supervised Panoptic Segmentation(2018), paper [[pdf](https://arxiv.org/abs/1808.03575)]

- [Panoptic Segmentation with a Joint Semantic and Instance Segmentation Network(2018)](https://deep-learning-study.tistory.com/862), paper [[pdf](https://arxiv.org/abs/1809.02110)]

- [Single Network Panoptic Segmentation for Street Scene Understanding(2019)](https://deep-learning-study.tistory.com/863), paper [[pdf](https://arxiv.org/abs/1902.02678)]

- [Panoptic Feature Pyramid Networks(2019)](https://deep-learning-study.tistory.com/867), paper [[pdf](https://arxiv.org/abs/1901.02446)]

- [IMP: Instance Mask Projection for High Accuracy Semantic Segmentation of Things(2019)](https://deep-learning-study.tistory.com/865), paper [[pdf](https://arxiv.org/abs/1906.06597)]

- [Object-Contextual Representations for Semantic Segmentation(2019)](https://deep-learning-study.tistory.com/894), paper [[pdf](https://arxiv.org/abs/1909.11065)]

- [CondInst, Conditional Convolution for Instance Segmentation(2020)](https://deep-learning-study.tistory.com/961), paper [[pdf](https://arxiv.org/abs/2003.05664)]

- Max-DeepLab, End-to-End Panoptic Segmentation wtih Mask Transformers, paper [[pdf](https://arxiv.org/abs/2012.00759)]

- [MaskFormer, Per-Pixel Classification is Not All You Need for Semantic Segmentation(2021)](https://deep-learning-study.tistory.com/940), paper [[pdf](https://arxiv.org/abs/2107.06278)]

- [Open-World Entity Segmentation(2021)](https://deep-learning-study.tistory.com/962), paper [[pdf](https://arxiv.org/abs/2107.14228)]

- Prompt based Multi-modal Image Segmentation(2021), paper [[pdf](https://arxiv.org/abs/2112.10003)]

- DenseCLIP, Language-Guided Dense Prediction with Context-Aware Prompting, paper [[pdf](https://arxiv.org/abs/2112.10003)]

- [Mask2Former, Masked-attention Mask Transformer for Universal Image Segmentation(2021)](https://arxiv.org/abs/2112.01527)

- [SeMask<, Semantically Masked Transformers for Semantic Segmentation(2021)](https://arxiv.org/abs/2112.12782)


## Semi-supervised Learning

- [Temporal ensembling for semi-supervised learning(2016)](https://deep-learning-study.tistory.com/757) , paper [[pdf](https://arxiv.org/abs/1610.02242)]

- [Mean teachers are better role models(2017)](https://deep-learning-study.tistory.com/758), paper [[pdf](https://arxiv.org/abs/1703.01780)]

- [Consistency-based Semi-supervised Learning for Object Detection(2019)](https://deep-learning-study.tistory.com/735), paper [[pdf](https://papers.nips.cc/paper/2019/hash/d0f4dae80c3d0277922f8371d5827292-Abstract.html)]

- [PseudoSeg, Designing Pseudo Labels for Semantic Segmentation(2020)](https://deep-learning-study.tistory.com/953), paper [[pdf](https://arxiv.org/abs/2010.09713)]

- [ReCo, Bootstrapping Semantic Segmentation with Regional Contrast(2021)](https://deep-learning-study.tistory.com/868), paper [[pdf](https://arxiv.org/abs/2104.04465)]

- [Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision(2021)](https://deep-learning-study.tistory.com/948), paper [[pdf](https://arxiv.org/abs/2106.01226)]

- [Soft Teacher(2021), End-to-End Semi-Supervised Object Detection with Soft Teacher](https://deep-learning-study.tistory.com/949), paper [[pdf](https://arxiv.org/abs/2106.09018)]

- [CaSP(2021), Class-agnostic Semi-Supervised Pretraining for Detection & Segmentation](https://deep-learning-study.tistory.com/960), paper [[pdf](https://arxiv.org/abs/2112.04966)]
