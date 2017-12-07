Environment: Python 3.6, OpenCV 3.3

Detection Part:

Put mser.py in any folder $ROOT, then create a folder $ROOT/test_detection, and put all the test images inside ROOT/test_detection, and run mser.py. Then you are expected to see Sample directories created with dict.txt containing possible grouped results, and detection.txt containing useful images for testing accuracy.
If you want to test mser.py on accuracy, please rename the test image as {number of icons or letters or digits}.{anything}.jpg, and mannually add number of correct detected letters/digits/icons to the third line of detection.py in every Sample{index} directory, and run accuracy.py

####################################################################

Eigenspace Part:
If you want just result, run 'runThisFile.m'
The name of test set should be ImageTest, which is already written in ‘runThisFile’.

I_mean.mat : trained mean vector of image matrix
max_distance.mat : maximum distance between original image and projected image
orthogonal_projection_matrix.mat : trained projection matrix onto eigen space

char_or_image2.m : calculate test image's distance with the space and write the result of detection. And return precisely detected number of character and wrongly detected number of icons.

runThisFile.m : This file will give you the result of detection using above files. And finally, calculate accuracy of eigen space.

eigenSpace.m : this file will not run if you directly run the 'runThisFile'. This file shows how the eigen space is made.

####################################################################

Recoginition part:
Run train.py to get the_train_model.pkl in $ROOT folder
Run testing.py to get accuracy of predicting characters. The testing data is in $ROOT/test,
