# How To Use BUT Before that:

### I know that my teammate will not read this so let me leave super important things at the top

### Please use processed_data.txt that I send to you/that you generate with writeInfo function (check the required file section)

## First time to learn
* change train.py main function's parameter from
```
'newTrain':True
```

* You can change the parameter of
  * epoch,
  * batch_size
  * train (file name for train)
  * test (file name for validation)
  * anchor coordinates (keep the number of anchors 5 **ONLY test.py**)
  * result_file_name (file name for output **ONLY test.py** )<br>
  I recommend you to change this because it shows the accuracies with the parameters.


## Second time to learn or later
* DON'T forget changing train.py main function's parameter back
```
'newTrain':False
```

## Output files

### train.py
* best_checkpoint.pth.tar<br>
if the current loss is the best, save it.

* checkpoint.pth.tar <br>
The latest loss

### test.py
* outImage folder<br>
automatically generate, it will contain the input images with bounding box

* result folder<br>
automatically generate, it will contain .txt file contains
  * argument (parameters) that you used in main function
  * result shwos the prediction of bounding box location
  ```
  name_of_test_image
  pred_x pred_y pred_width pred_height accuracy
  ```


## How to use
Change the argument of main function in train.py file and run
```
!pyton train
```

if you wanna call train/test from the other python file, set the argument like in the main function and call
```
from train import train
train(arg)
```


## Other things that you don't really need to read



1. Required modules
  * openCV
  * pytorch

  Also, the leaning system and models are designed for gpu

2. Required files
  * Resized images (416x416) for train/tes<br>
    To get them, run the following function rescale the images
    ```
    from label_resize import resize
    resize("Dataset/Color/*.jpg")
    ```

  * Under Dataset folder<br>

    format: ```name_of_image xmin ymin width height```

    This txt is for training/testing, and can be clipped by existing file<br>
    e.g. export from line 101 to 300, save in ```small_train.txt```
    ```
    sed -n '101, 300p' processed_data.txt > small_train.txt
    ```
    Otherwise, you use the function (I slightly changed your code for image name) to generate the file from json file by
    ```
    from label_resize import resize
    writeInfo("Dataset/annotation.json")
    ```
    This generate the whole image information (left and right had separately), so if you lost ```processed_data.txt``` that I gave you, please use this.






  * weight (if test)
    * use ```best_checkpoint.pth.tar```

3. argument/parameters<br>
explained above

4. output files
explained above
