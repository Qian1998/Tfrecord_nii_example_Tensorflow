## About this code  
This code is written for feeding the data to Neural Network (TF) using **tfrecords**.
Specifically, **It is a example for medical case (LiTs)** **(*.nii)**
  
### Reference  
1. http://warmspringwinds.github.io/  
2. http://bcho.tistory.com/1190  
3. https://stackoverflow.com/questions/34050071/tensorflow-random-shuffle-queue-is-closed-and-has-insufficient-elements/43370673  
4. https://github.com/wjcheon/Tfrecord_example_Tensorflow (self-citation)  
5. 2017 MICCAI Liver Tumour Segmentation Challenge. (LiTS) 2017 MICCAI Liver Tumour Segmentation Challenge. (LiTS)   

### Please read this phragraph
Medical image (*.nii format) was uploaded in this Github. 
But, I compressed Medical data (*.nii) and then it was splited under 100 MB due to limitation of Github.  
Follwing this commend for **Join** and **Uncompress**  
>> cd Tfrecord_nii_example_Tensorflow/  
>> cd Tfrecord_Selialization/  
>> cat ./data_compress.tar.gz.part* > medical_data.tar.gz  
>> tar -xvzf medical_data.tar.gz   

### Figure    
<img src = https://github.com/wjcheon/Tfrecord_nii_example_Tensorflow/blob/master/TFrecord_Parse/tfrecorad_nii_sample_01.png />  
Fig.1 Parsing and Imshow using Tfrecord    
  
        
date: 2018.03.15  
All rights of this code reserved to **Danill** and **Wonjoong Cheon** with **SSoo**  
the code was update and fix to perform at Tensorflow version 1.6.  
If you have any question for this code, please send the e-mail to me.  
   


## Who am I 
**Wonjoong Cheon**  
Ph.D intergrated program  
Medical Physics Lab. - SUMP Lab.  
Samsung Advanced Institute for Health Science & Technology(SAIHST), SungKyunKwan University.  
B.E. Dept. of Information and Communication Engineering , Yonsei University.  
B.S. Dept. of Radiological Science, Yonsei University.  

Laboratry
Samsung Medical Center (Medical Physics)  
National Cancer Center (Computer Vision: 3D vision)  
Vatech Vision reasearch Center (CT reconstruction algorithm)  

**Interest field**  
Medical physics, Monte-carlo simulation, Machine learning  

wonjoongcheon@gmail.com,   
Samsung Medical Center (06351) 81 Irwon-ro Gangnam-gu, Seoul, Korea
