# CNN_Image_Annotaion
Implementation of some basic Image Annotation methods (using various loss functions & threshold optimization) on Corel-5k dataset with PyTorch library

## Dataset
<div align="justify"> There is a 'Corel-5k' folder that contains the (Corel-5k) dataset with 5000 real images (and 13,500 fake images) from 50 categories. Each category includes 100 images, and there are 260 labels totally in the vocabulary. This is one of the benchmark datasets in the image annotation field, which the low number of data and diversity of labels' distribution makes it challenging. Additionally, There are also datasets with more image data, such as (IAPR TC-12) with 19,627 photos and (ESP-GAME) with 20,770 photos that are commonly used in this field. <br>
Usually, (Corel-5k) is divided into 3 parts: a training set of 4000 images, a validation set of 500 images, and a test set of 499 images. In other words, the total number of images for training is 4500 (18,000 wih fake images) and for validation is 499. </div> <br>

Specific information of the three datasets is provided in tabel below:
*Dataset* | *Number of images* | *Vocabulary size* | *Training images*  | *Testing images*  | *Words per img*  | *Img per word* 
------------ | ------------- | ------------- | ------------- | ------------ | ------------- | ------------- |
Corel-5k | 5000  | 260 | 4500 | 499 | 3.4 | 58.6 
ESP-Game | 20,770  | 268 | 18,689 | 2081 | 4.7  | 362.7 
IAPR TC-12 | 19,627  | 291 | 17,665 | 1962 | 5.7  | 347.7 

You can see the distribution of some labels below: (total 5000 images)
class name | count
------------ | -------------
sails     | 2   
orchid    | 2   
butterfly | 4   
cave      | 6   
...       | ... 
cars      | 151 
flowers   | 296 
grass     | 497 
tree      | 947 
sky       | 988 
water     | 1120

There is also an 'unlabeled images' folder with 10 photos without any label. You can evaluate the accuracy of your model on these random images or any other images with the same content.

## Data Augmentation
<div align="justify"> As I mentioned previously, (Corel-5k) has only 4500 images as training data, which makes it impossible to train with complicated models and results in overfitting. To overcome this issue, data augmentation methods could be effective. As mentioned in the paper of Xiao Ke, et al, generative adversarial networks are one of the best models for data augmentation. <br> 
They proposed a multi-label data augmentation method based on Wasserstein-GAN. (The process of ML-WGAN is shown in the picture below). Due to the nature of multi-label images, every two images in a common dataset usually have different numbers and types of labels, therefore, it is impossible to directly use WGAN for multi-label data augmentation. The paper suggested using only one multi-label image at a time since the noise (z) input by the generator can only be approximated by the distribution of that image iteratively. As the generated images only use one original image as the real data distribution, they all have the same number and type of labels and have their own local differences while the overall distributions are similar. <br>
There is a 'DataAugmentation' folder that contains the codes of ML-WGAN, which is similar to the paper "Improved Training of Wasserstein GANs". because of the fact that one original image has to be used as real data distribution, I trained the network for each image individually and generated 3 more images for every original image, which increased the size of the training images to 18,000. </div> <br>
  
An example of the generated images:
![example](https://user-images.githubusercontent.com/85555218/164970970-bddb3244-97b7-4652-a09a-50691a5d80d4.png)
  
**ML-WGAN**
![WGAN](https://user-images.githubusercontent.com/85555218/164969445-0d175b0d-8018-435d-b9cf-7d03f756b940.png)

 **Generator**
![generator](https://user-images.githubusercontent.com/85555218/169645470-0a977c2f-93c4-4360-a2f8-a17654b89365.png)

 **Critic**
![discriminator](https://user-images.githubusercontent.com/85555218/169645474-79af2d4c-dad3-45e6-acc9-14518bb331cc.png)
  
## Convolutional models
<div align="justify"> Various convolutional models have been used in diverse tasks, but I chose ResNet, ResNeXt, Xception, and TResNet, which have shown good results in recent studies. Due to the pre-training of all of these models, there is no need to initialize weights. By comparing the results obtained from the four mentioned models, I will determine which model is the most effective. </div> <br>

The images below show the structure of these models:
![info](https://user-images.githubusercontent.com/85555218/178442133-5c40cedd-bb81-495d-bec0-b6d5a9be51a6.png)

**Xception**
![Xception](https://user-images.githubusercontent.com/85555218/178439013-94a68bab-a2a6-4881-99c3-99bf24aab0b2.png)
number of trainable parameters: 21,339,692
  
**ResNeXt50**
![ResNeXt50](https://user-images.githubusercontent.com/85555218/178438991-d7d78f74-0deb-4f9a-9990-91459b401a2a.png)
number of trainable parameters: 23,512,644

**TResNet-m**
![TResNet_m](https://user-images.githubusercontent.com/85555218/184323915-a5aaf429-81ad-4d1a-b8c7-8ea6f6ac6239.png)
number of trainable parameters: 29,872,772

**ResNet101**
![ResNet101](https://user-images.githubusercontent.com/85555218/178438970-99566704-fc24-4ec7-ad65-d4c57d8e1055.png)
number of trainable parameters: 43,032,900
  
## Evaluation Metrics
<div align="justify"> Precision, Recall, and F1-score are the most popular metrics for evaluating CNN models in image annotation tasks. I've used per-class (per-label) and per-image (overall) precision, recall, and f1-score, which are common in image annotation papers. </div> <br>

The aforementioned evaluation metrics formulas can be seen below:
![evaluation-metrics](https://user-images.githubusercontent.com/85555218/178450245-945e6802-8ca6-44e1-9d32-6c26a8fb7423.png)

Another evaluation metric used for datasets with large numbers of tags is N+:
![N-plus](https://user-images.githubusercontent.com/85555218/178450305-61aa0515-9d3f-4e52-8a78-cb5066176ab6.png)

> Note that the per-class measures treat all classes equal regardless of their sample size, so one can obtain a high performance by focusing on getting rare classes right. To compensate this, I also measure overall precision/recall which treats all samples equal regardless of their classes.

## Loss Functions & Thresholding
<div align="justify"> I've used several loss functions and thresholding methods to compare their results on the models mentioned above. Classifications with multi-label are typically converted into multiple binary classifications. Based on the number of labels, models predict the logits 𝑥_𝑖 of the 𝑖-th label independently, then the probabilities are given by normalizing the logits with the sigmoid function as 𝑝_𝑖 = 𝜎(𝑥_𝑖). Let 𝑦_𝑖 denote the ground-truth for the 𝑖-th label. (Logits are interpreted to be the not-yet normalized outputs of a model). </div> <br>

The binary classification loss is generally shown in the image below:
![binary-classification-loss](https://user-images.githubusercontent.com/85555218/178476706-2689a73c-0866-4bfe-ac6c-c41d0ab63d24.png)

### 1: binary cross entropy loss + (fixed threshold = 0.5)
The binary cross entropy (BCE) loss function is one of the most popular loss functions in multi-label classification or image annotation, which is defined as follows for the 𝑖-th label:
![BCE](https://user-images.githubusercontent.com/85555218/178477886-4e7c642a-95f2-4e0f-be9a-30f9739dad63.png)

### results 
best model | global-pooling | batch-size | num of training images | image-size | optimizer | epoch time
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | -------------
**TResNet-m** | avg | 32 | 4500 | 448 * 448 | Adam | 135s
  
data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.994 | 0.989 | 0.992 
*testset* per-image metrics | 0.726 | 0.589 | 0.650
*trainset* per-class metrics | 0.995 | 0.985 | 0.990
*testset* per-class metrics | 0.453 | 0.385 | **0.416**

data | N+ 
------------ | ------------- 
*trainset* | 260
*testset* | 147
  
### threshold optimization with matthews correlation coefficient
<div align="justify"> The parameters of the convolutional network will be fixed when the training is complete, then we calculate MCC separately for each label of training data with these values of the thresholds: [0.05 - 0.1 - 0.15 - 0.2 - 0.25 - 0.3 - 0.35 - 0.4 - 0.45 - 0.5 - 0.55 - 0.6 - 0.65 - 0.7]. finally the threshold with the best MCC will be chosen for that label. Experiments showed that labels with a low frequency have lower thresholds and vice versa. </div> <br>

The following picture illustrates the MCC formula:
![MCC](https://user-images.githubusercontent.com/85555218/178485429-455ba1e5-d2f8-4c77-975b-87d836810d68.png)

> Matthews correlation coefficient calculates the correlation between the actual and predicted labels, which produces a number between -1 and 1. Hence, it will only produce a good score if the model is accurate in all confusion matrix components. MCC is the most robust metric against imbalanced dataset issues.

### results 
best model | global-pooling | batch-size | num of training images | image-size | optimizer | epoch time
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | -------------
**TResNet-m** | avg | 32 | 4500 | 448 * 448 | Adam | 135s

data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.994 | 0.989 | 0.992 
*testset* per-image metrics | 0.726 | 0.589 | 0.650
*trainset* per-class metrics | 0.995 | 0.985 | 0.990
*testset* per-class metrics | 0.453 | 0.385 | 0.416
*testset* per-class metrics + MCC | 0.445 | 0.451 | **0.448**

data | N+ 
------------ | ------------- 
*trainset* | 260
*testset* | 147
*testset* + MCC | 164

### 2: focal loss
<div align="justify"> The difference betwen Focal Loss and BCE loss is that Focal Loss makes it easier for the model to predict labels without being 80-100% sure that this label is present. In simple words, giving the model a bit more freedom to take some risks when making predictions. This is particularly important when dealing with highly imbalanced datasets. <br>
BCE loss leads to overconfidence in the convolutional model, which makes it difficult for the model to generalize. In fact, BCE loss is low when the model is absolutely sure (more than 80% or 90%) about the presence and absence of the labels. However, as seen in the following picture when the model predicts a probability of 60% or 70%, the loss is lower than BCE. </div>

![focalloss-pos](https://user-images.githubusercontent.com/85555218/178502431-9c03a4d1-89e1-4ba7-b4bf-1ad9b4953a00.png)

The focal loss formula for the 𝑖-th label is shown in the image below:
![focalloss](https://user-images.githubusercontent.com/85555218/178492084-c5e3f1e8-cff7-4664-88ff-42bb2133692a.png)

**<div align="justify"> To reduce the impact of easy negatives on multi-label training, we use focal loss. However, setting high 𝛾 may eliminate the gradients from rare positive labels. As a result, we cannot expect a higher recall if we increase the value of 𝛾. I will elaborate on this issue in the [Gradient Analysis](https://github.com/parham1998/CNN_Image_Annotaion#gradient-analysis) section. </div>**

### results   
best model | global-pooling | batch-size | num of training images | image-size | optimizer | epoch time | 𝛾
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
**TResNet-m** | avg | 32 | 4500 | 448 * 448 | Adam | 135s | 3
  
data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.992 | 0.982 | 0.987
*testset* per-image metrics | 0.758 | 0.581 | 0.658
*trainset* per-class metrics | 0.995 | 0.978 | 0.987
*testset* per-class metrics | 0.452 | 0.366 | 0.405
*testset* per-class metrics + MCC | 0.483 | 0.451 | **0.466**

data | N+ 
------------ | ------------- 
*trainset* | 260
*testset* | 139
*testset* + MCC | 162

### 3: asymmetric loss
<div align="justify"> As I mentioned <a href="https://github.com/parham1998/CNN_Image_Annotaion#dataset">here<a>, there is a huge imbalance in the distribution of labels in the (Corel-5k) and other annotation datasets. The train set might contain labels that appear only once, as well as labels that appear more than 1,000 times. Unfortunately, due to the nature of annotation datasets, there isn't anything that can be done to overcome this problem. <br> 
But, there is another imbalance regarding the number of positive and negative labels in a picture. In simple words, most multi-label pictures contain fewer positive labels than negative labels (for example, each image in the (Corel-5k) dataset contains on average 3.4 positive labels). </div>

![imbalance labels](https://user-images.githubusercontent.com/85555218/178682215-fde04a7e-16b6-4363-b824-bb140aa37dab.png)
<div align="justify"> In training, this imbalance between positive and negative labels dominates the optimization process, which results in a weak emphasis on gradients from positive labels (more information in the <a href="https://github.com/parham1998/CNN_Image_Annotaion#gradient-analysis">Gradient Analysis<a> section).
Asymmetric loss operates differently on positive and negative labels. It has two main parts: <br>
<strong> 1. Asymmetric Focusing </strong> <br>
Unlike the focal loss, which considers one 𝛾 for positive and negative labels, positive and negative labels can be decoupled by taking 𝛾+ as the focusing level for positive labels, and 𝛾− as the focusing level for negative labels. Due to the fact that we are seeking to emphasize the contribution of positive labels,  we usually set 𝛾− > 𝛾+. <br>
<strong> 2. Asymmetric Probability Shifting </strong> <br>
Asymmetric focusing reduces the contribution of negative labels to the loss when their probability is low (soft thresholding). However, this attenuation is not always sufficient due to the high level of imbalance in multi-label classifications. Therefore, we can use another asymmetric mechanism, probability shifting, which performs hard thresholding on very low probability negative labels, and discards them completely. The shifted probability is defined as 𝑝_𝑚 = max⁡(𝑝 − 𝑚, 0), where the probability margin 𝑚 ≥ 0 is a tunable hyperparameter. </div> <br>

In the image below, the asymmetric loss formula for the 𝑖-th label can be seen:
![asymmetricloss](https://user-images.githubusercontent.com/85555218/178691710-8339cbbc-fc27-4e25-96dd-7b70091db713.png)

### results   
best model | global-pooling | batch-size | num of training images | image-size | optimizer | epoch time | 𝛾+ | y- | m
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
**TResNet-m** | avg | 32 | 4500 | 448 * 448 | Adam | 141s | 0 | 4 | 0.05
  
data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.956 | 0.999 | 0.977
*testset* per-image metrics | 0.624 | 0.688 | 0.654
*trainset* per-class metrics | 0.939 | 0.999 | 0.968
*testset* per-class metrics | 0.480 | 0.522 | 0.500
*testset* per-class metrics + MCC | 0.473 | 0.535 | **0.502**

data | N+ 
------------ | ------------- 
*trainset* | 260
*testset* | 179
*testset* + MCC | 184
  
<div align="justify"> In addition to dealing with negative labels via ASL, it is possible to take advantage of multi-label learning by delicately tailoring the loss for positives. In fact, multi-label learning does not often use hard-mining methods (recognizing hard labels with a low probability) that can yield improvement in single-label learning. For example, ASL with 𝛾+ = 0 only adopts the naive BCE loss for positives. Or increasing the value of 𝛾+ turn the positive side of the ASL loss function into the focal loss. <br>
The <a href="https://github.com/parham1998/CNN_Image_Annotaion#gradient-analysis">Gradient Analysis<a> section indicates that excessively hard mining brought by focal loss is inappropriate for multi-label learning. Specifically, despite the decrease in the number of hard positives (low probability), focal loss only pushes the very small proportion of hard positives to a high probability. Meanwhile, Focal loss ignores a large proportion of semi-hard ones with a moderate probability. Therefore, seeking to further emphasize semi-hard positives compared with Focal loss can be effective. <br>
As an intuitive method, a margin can be subtracted from the logits. Since they are smaller values, they are given larger gradients due to the feature of sigmoid function. </div> <br>

The updated asymmetric loss formula for the 𝑖-th label can be seen in the picture below:
![asl_new](https://user-images.githubusercontent.com/85555218/182120924-3cc3c705-3f1c-46f2-a904-d1389e595127.png)
  
### results   
best model | global-pooling | batch-size | num of training images | image-size | optimizer | epoch time | 𝛾+ | y- | m1 | m2 | th
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------
**TResNet-m** | avg | 32 | 4500 | 448 * 448 | Adam | 135s | 3 | 4 | 1.1 | 0.05 | 0.25
  
data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.942 | 0.999 | 0.970
*testset* per-image metrics | 0.594 | 0.717 | 0.649
*trainset* per-class metrics | 0.909 | 1.000 | 0.952
*testset* per-class metrics | 0.466 | 0.554 | **0.506**
*testset* per-class metrics + MCC | 0.484 | 0.563 | **0.5207**

data | N+ 
------------ | ------------- 
*trainset* | 260
*testset* | 189
*testset* + MCC | 191
  
The result of the trained model with asymmetric loss + MCC on one batch of test data:
![AS_results](https://user-images.githubusercontent.com/85555218/182779841-7040c0d7-2bb2-4ad9-aeb0-5c42e6aa119f.jpg)
  
### 4: log-sum-exponential-pairwise loss
<div align="justify"> LSEP loss had been proposed in the paper of Y. Li, et al, and was an improvement for the simple pairwise ranking loss function. In fact, LSEP is differentiable and smooth everywhere, which makes it easier to optimize. </div>

![LSEP](https://user-images.githubusercontent.com/85555218/178549837-9d4780bd-3f8e-40b3-ba5a-5df0a5c9a800.png)

### results 
best model | global-pooling | batch-size | num of training images | image-size | optimizer | epoch time
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | -------------
**ResNeXt50** | avg | 32 | 4500 | 224 * 224 | Adam | 45s

data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.639 | 0.975 | 0.772
*testset* per-image metrics | 0.490 | 0.720 | 0.583
*trainset* per-class metrics | 0.576 | 0.914 | 0.706
*testset* per-class metrics | 0.403 | 0.548 | **0.464**

data | N+ 
------------ | ------------- 
*trainset* | 255
*testset* | 188
  
The result of the trained model with LSEP loss on one batch of test data:
![lsep_results](https://user-images.githubusercontent.com/85555218/169834100-b3d22834-f2bf-4090-add2-28d97b4c2d7e.png)

## Gradient Analysis
<div align="justify"> The purpose of this section is to analyze and compare different losses' gradients, to give us a better view of the properties and behavior of losses mentioned above. Since in practice, weights of the network are updated according to the gradient of the loss, based on the input logit 𝑥, it is useful to look at the gradients. <br> </div>
The loss gradients for positive and negative labels in BCE Loss are: 

![BCE](https://user-images.githubusercontent.com/85555218/181238934-99629eb2-ddf2-4067-9e4a-e04691b44f6b.png)
  
<div align="justify"> As I mentioned before, the massive imbalance between positive and negative labels in an image affects the optimization problem. And due to its symmetrical nature (every positive and negative label contributes equally), BCE cannot overcome this problem. 
In the image above, the red line indicates that BCE loss always has a gradient greater than zero, even for those negative labels whose probability (p) is close to zero. During training, this causes gradients from positive labels to tend to be underemphasized due to the high number of negative labels present in annotation problems. <br> 
To resolve this issue, we can either <strong> reduce the contribution of negative labels to the loss </strong> or <strong> increase the contribution of positive labels to the loss</strong>. </div> <br>
  
<div align="justify"> Another symmetric loss function that tries to down-weight the contribution from easy negatives is focal loss. you can see this loss gradients for positive and negative labels in the picture below: </div>

![focal](https://user-images.githubusercontent.com/85555218/181277787-215bad8e-75e6-45e4-9f5c-52be365ac130.png)
  
<div align="justify"> As can be seen in the image above, the focal loss is successful in reducing the contribution from easy negatives (the loss gradient for low probability negative labels is near zero), but due to its symmetrical nature, focal loss also eliminates the gradients from the rare positive labels. <br>
The loss gradient for positive labels indicates that it only pushes a very small ratio of hard positives to a high probability and ignores a large ratio of semi-hard ones with a medium probability. <br>
The contribution of easy negative labels would decrease more when 𝛾 is increased, but on the other hand gradients of more positive labels would disappear. </div>
  
<div align="justify"> In order to overcome the problems associated with BCE loss and focal loss, as I mentioned previously asymmetric loss is one of the best solutions. The reason why the asymmetric loss has a significant effect on the result can be seen in the illustration of loss gradients for positive and negative labels: </div>
  
![AS](https://user-images.githubusercontent.com/85555218/181906650-45b53250-19dc-4a2c-bf75-def3746b9bc5.png)
  
<div align="justify"> One of our objectives was to reduce the contribution of negative labels to the loss, but symmetric loss functions such as focal loss  did not have the ability to keep the contribution of positive labels at the same time as reducing the contribution of negative labels. However, by choosing 𝛾− and 𝛾+ differently, the objective can be achieved as shown in the image above. Furthermore, the loss gradients for negative labels indicate that hard thresholding (m) not only causes very low probability negative labels ignored completely, but also affects the very hard negative labels, which are considered missing labels. As a definition of missing labels, we can say that if the network computes a very high probability for a negative label, then the label may be positive. <br>
The loss gradient value for the very small portion of negative labels with a large probability (p > 0.9) is very low, which shows that if the network is absolutely sure about a negative label, it can accept it as a missing label. </div> <br>
  
<div align="justify"> The loss gradient of the previous method show that asymmetric loss functions can lead to a reduction in the contribution of the negative labels to the loss function. however, the contribution of the positive labels still remains unchanged. <br>
As I said before, because of the massive difference between the number of positive and negative labels in an image, increasing the contribution of positive labels could be valuable in image annotation datasets. <br>
The image below shows how a very small change in the sigmoid function, which we have seen in the last paragraph of the asymmetric loss section, gives us the ability to control the contribution of positive labels to the loss gradient: </div>
  
![AS_positive](https://user-images.githubusercontent.com/85555218/182641503-aad9c1ca-97f6-4cd9-8cd9-b6a45ad5b2c1.png)
  
<div align="justify"> In fact, in the training process, penalizing positive labels that have been correctly predicted is not necessary. But in annotation datasets, that only include images with 4 or 5 positive labels and plenty of negative labels, to grow the contribution of positive labels we have to increase the amount of the loss even for these rightly predicted labels. <br>
The above loss gradient for positive labels has two parts: <br> 
The first part contains the loss gradient for positive labels that have been predicted wrongly. It is clear that we should penalize these labels much more than correctly predicted labels. <br>
The second part includes the loss gradient for positive labels that have been predicted rightly. But as I mentioned about image annotation Challenges, we have to increase (not as much as incorrectly predicted labels) the loss gradient even for these labels.
</div>
  
## Conclusions
<div align="justify"> To sum up, different types of convolutional models and loss functions gave me different results. between convolutional models, TResNet performed better than the other models not only in the result but also in memory usage. Based on the results, LSEP loss leads to an increase in recall value, so the model predicts more labels per image. However, BCE and focal loss increase precision, so the model is more cautious in predicting labels, and will try to predict more probable labels. The best results from both f1-score and N+ were obtained by the model which optimized by asymmetric loss functions, and this shows the superiority of this type of loss function compared to other loss functions. <br>
In order to compare the results, I have tried many experiments including changing the resolution of the images (from 224 * 224 to 448 * 448), changing the global pooling of the convolutional models (from global average pooling to global maximum pooling), etc. Among these experiments, the aforementioned results are the best. <br>
unfortunately, the data augmentation method (ML-WGAN) did not produce the expected results, and could not improve the overfitting problem. <br>
In this project, I used a variety of CNNs and loss functions without taking label correlations into account. By using methods such as graph convolutional networks (GCNs) or recurrent neural networks (RNNs) that are able to estimate label correlations, better results may be obtained. </div>
  
## References
Y. Li, Y. Song, and J. Luo. <br>
*"Improving Pairwise Ranking for Multi-label Image Classification"* (CVPR - 2017)

T. Ridnik, E. Ben-Baruch, N. Zamir, A. Noy, I. Friedman, M. Protter, and L. Zelnik-Manor. <br>
*"Asymmetric Loss For Multi-Label Classification"* (ICCV - 2021)

I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. <br>
"Improved Training of Wasserstein GANs" (arXiv - 2017)
