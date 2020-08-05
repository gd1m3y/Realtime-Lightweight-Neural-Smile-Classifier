# SMILE DETECTOR
smile detector is a light weight smile detection model which can predict whether a person is smiling or not
# Work Flow
The work flow is as follows
![img_work_flow](https://github.com/gd1m3y/SMILE-DETECTION/blob/master/workflow.png?raw=true)
* Preprocessing -the image is processed to the requirements so that it can be fed to the model
* face detection - open cvs casscade classifier is used to detect the face in the image
* smile detection - after detection of the face that part is fed to the model for smile detection
# Model Architecture 
LE-NET
![img_work_flow](https://github.com/gd1m3y/SMILE-DETECTION/blob/master/lenet.png?raw=true)
#  Features

  - Light weight
  - Very easy to use and customize
  - Easy to re-train

### Tech

SMILEs uses a number of open source projects to work properly:

* [Python] - Python 3.6 the mother of data science languages
* [Pycharm] - awesome python editor
* [Open-Cv] - Open-cv version 4.3.3
* [tensorflow 2.X] - a deep learning framework
### Installation

Chat bot 0  requires python 3.6 with following libraries:
* tensorflow-2.x 
* Open cv
* numpy
* immutils

if you are using conda env you can create a conda enviorment with python 3.6
```sh
conda create env python ==3.6
pip install tensorflow

```




### Development

Want to contribute? Great!

contact me at narayanamay123@gmail.com
### Todos

 - improve accuracy since it requires us to get very close to the camera
 - Re-train the model in a better data set
License
----

MIT
