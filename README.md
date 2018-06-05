# EE267-Final-Project
Final Project for EE267, Virtual Reality.

Project is on creating an eye tracking algorithm using CNNs.

Saved data is too large for github, so you will have to save the data yourself
and train the models yourself. Don't worry, it's pretty easy. If you have access
to our Google Drive though, you will not need to worry about it as you will have
the saved data and models.

Make sure you are in the Scripts folder when running the code below.

Before starting, make sure you have python installed on your computer, as well
as these following packages in python:

    pickle, os, numpy, operator, cv2, scipy, torch, torchvision, random, matplotlib

Once you have installed all of the packages, type python into the terminal and
follow what's below.

If you do not have access to Google Drive, run the following in order:
```python
  import preprocess
  import model
  import interp

  preprocess.gatherData()
  preprocess.saveDistance()
  preprocess.createUniqueYs(mode='save')
  preprocess.setup(mode='save')

  model.testHyperParameters('train', False, 'adam')
  model.testHyperParameters('train', False, 'rmsprop')
  model.testHyperParameters('train', False, 'sgd')
  model.testHyperParameters('train', True, 'adam')
  model.testHyperParameters('train', True, 'rmsprop')
  model.testHyperParameters('train', True, 'sgd')

  model.testHyperParameters('test', False, 'adam')
  model.testHyperParameters('test', False, 'rmsprop')
  model.testHyperParameters('test', False, 'sgd')
  model.testHyperParameters('test', True, 'adam')
  model.testHyperParameters('test', True, 'rmsprop')
  model.testHyperParameters('test', True, 'sgd')

  interp.interp("../Models/model_adam_False_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_rmsprop_False_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_sgd_False_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_adam_True_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_rmsprop_True_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_sgd_True_0.001_0.9_(0.5, 0.999)_0.9")
```

If you do have access to GoogleDrive, run the following:
```python
  import model
  import interp

  model.testHyperParameters('test', False, 'adam')
  model.testHyperParameters('test', False, 'rmsprop')
  model.testHyperParameters('test', False, 'sgd')
  model.testHyperParameters('test', True, 'adam')
  model.testHyperParameters('test', True, 'rmsprop')
  model.testHyperParameters('test', True, 'sgd')

  interp.interp("../Models/model_adam_False_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_rmsprop_False_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_sgd_False_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_adam_True_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_rmsprop_True_0.001_0.9_(0.5, 0.999)_0.9")
  interp.interp("../Models/model_sgd_True_0.001_0.9_(0.5, 0.999)_0.9")
```
