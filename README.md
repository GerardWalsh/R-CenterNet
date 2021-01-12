# R-CenterNet
Detector for rotated-object based on CenterNet

### preface
The original intention of this work is to provide a **extremely compact** code of CenterNet and detect rotating targets: 1.0
 ~~~
  ${R-CenterNet_ROOT}
  |-- backbone
  `-- |-- dlanet.py
      |-- dlanet_dcn.py
  |-- Loss.py
  |-- dataset.py
  |-- train.py
  |-- predict.py
 ~~~
At the request of readers, 2.0 was subsequently updated：2.0
 ~~~
  ${R-CenterNet_ROOT}
  |-- labelGenerator
  `-- |-- Annotations
      |-- voc2coco.py
  |-- evaluation.py
 ~~~
 2.0 and the data/airplane, imgs, ret folders are not required. If you just want to get started quickly, 1.0 is enough。

#### demo
* R-DLADCN(this code)(How to complie dcn refer to the original code of [CenterNet](https://github.com/xingyizhou/centernet))
    * ![image](ret/R-DLADCN.jpg)
* R-ResDCN(just replace cnn in resnet with dcn)
    * ![image](ret/R-ResDCN.jpg)
* R-DLANet(not use dcn if you don't know how to complie dcn)
    * ![image](ret/R-DLANet.jpg)
* DLADCN.jpg
    * ![image](ret/DLADCN.jpg)

#### notes
 * Refactored the original [code](https://github.com/xingyizhou/centernet) to make code more concise.
 * To complie dcn and configure the environment, refer to the original code of [CenterNet](https://github.com/xingyizhou/centernet).
 * For data processing and more details, refer to [here](https://zhuanlan.zhihu.com/p/163696749)
 * torch version==1.2，don't use version==0.4!
#### train your data
 * label your data use labelGenerator;
 * modify all num_classes to your classes num, and modify the num of hm in your back_bone, such as:
   def DlaNet(num_layers=34, heads = {'hm': your classes num, 'wh': 2, 'ang':1, 'reg': 2}, head_conv=256, plot=False):

#### Related projects
* [R-CenterNet_onnxruntime](https://github.com/ZeroE04/R-CenterNet_onnxruntime)【C++】
* [CenterNet](https://github.com/xingyizhou/centernet)
