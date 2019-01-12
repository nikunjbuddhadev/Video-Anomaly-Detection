# Video-Anomaly-Detection
Source - http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html

A major aspect of the project work was to develop a proper classification method and an efficient algorithm for the purpose of the detection of the fall activity. 
Anomaly detection is a technique used to identify unusual patterns that do not conform to expected behavior, called outliers. Given a dataset of frames of videos of daily activity living and fall sequences in which prediction was made to detect an abnormal activity (falling person) from a test set. 
Frames given were used to form dynamic images for all the videos using MATLAB. Those dynamic images were further processed in python(code attached). In further processing, Histogram of Oriented Gradients(HOG) features from each dynamic image were extracted which are returned in a 1-by-n vector. The returned features encode local shape information from regions within an image. These vectors from each image were used to form a dataframe which was then used as an input to our neural network for the detection of fall activity. 

