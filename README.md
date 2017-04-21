[resnet]: images/resnet.png "ResNet Model"
[vggnet]: images/vggnet.png "VggNet Model"
[vggnet5]: images/vggnet5.png "VggNet5 Model"

# MNIST Competition Tensorflow KR Group

* MNIST competition submission files
* Used Keras
* [Model Architecture](#model-architectures)
    * [ResNet](#resnet)
    * [VggNet](#vggnet)
    * [VggNet5](#vggnet5)

## Performance

|  **Model**  |                 **Description**                 | **Accuracy** |
|:-----------:|:-----------------------------------------------:|:------------:|
|   VGG-like  |             VGGNet-like but smaller             |    99.71%    |
| Resnet-like |             ResNet-like but smaller             |    99.60%    |
|   VGG-like  | VGGNet-like but even smaller than the first one |    99.63%    |
|  **Final**  |          **Ensemble 3 models + Voting**         |  **99.80%**  |

## Run

#### Evaluation
```bash
python evaluation.py
```

#### Train
```bash
python resnet.py 10 # 10 epochs & resnet
python vgg16.py 10 # 10 epochs & vgg
python vgg5.py 10 # 10 epochs & vgg
```

## File descriptions
```bash
├── evaluation.py # evaluation.py
├── images # model architectures
│   ├── resnet.png
│   ├── vggnet5.png
│   └── vggnet.png
├── MNIST # mnist data (not included in this repo)
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   └── train-labels-idx1-ubyte.gz
├── model # model weights
│   ├── resnet.h5
│   ├── vggnet5.h5
│   └── vggnet.h5
├── model.py # base model interface
├── README.md
├── utils.py # helper functions
├── resnet.py
├── vgg16.py
└── vgg5.py
```


## Model Architectures

#### ResNet
![resnet][resnet]

#### VggNet
![vggnet][vggnet]

#### VggNet5
![vggnet5][vggnet5]