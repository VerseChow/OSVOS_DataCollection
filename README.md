# OSVOS_DataCollection
## Data collection via OSVOS method in DAVIS Challenge

The use of this data collection tool is for automatically annotate bounding box in each image for object detection. And it will save the annotation and image into a Pascal VOC folder structure like dataset. The format of the annotation is like Pascal VOC dataset.

The main idea came from [OSVOS: One-Shot Video Object Segmentation](https://arxiv.org/pdf/1611.05198.pdf)

Input: Vedio Frames, First Frame Ground Truth Segmentation

Output: Segmentation for Each Frame, Bounding Box information, Pascal VOC folder structure like dataset

## Pipeline
1. Train with Feed first frame into FCN network to extract the feature of the object wanted to be segmented until loss get down a threshold
2. Feed in the rest vedio frames to get segmentation of the object and Bounding Box information of the object.

We have a demo video on youtube, please click link: [OSVOS Data Collection](https://www.youtube.com/watch?v=xKtegsclTI8)

## Usage

There are two script files to run this tool. Please run the script file one by one. The first one is oneshot_training.sh, the second one is data_collection.sh.

Before run the scripts, please do following things.

### oneshot_training.sh
1. Inculde pretrained model into pretrained_checkpoint folder. For the model please contact me (ywchow@umich.edu)
2. In the dataset you want to train, please specify the folder structure like following:
```
	<Folder Name>
	|*.jpg	     
	|	gt 
	|	|*.png 
```
3. image in gt folder is the groundtruth segmentation corresponding to the jpg image the previous level of folder.
4. Please install [GIMP Image Editor](https://www.gimp.org/) tool. You could use it to get get groundtruth. You could insert the .jpg image and manually cropping out the groundtruth segmentation and save it in gt folder as a .png image.
5. The model after oneshot_training will be saved in checkpoint folder.
6. The example to use oneshot_training.sh is like: ./oneshot_training.sh ./table/table_9 "001". The first argument is dataset directory to train, the second argument is name of the picture to train.

### data_collection.sh
1. The example to use data_collection.sh is like: ./data_collection.sh ./table/table_9 progress. The first argument is dataset directory to collect, and the second argument will be the name of the output well structured dataset.
2. The script will create a folder in the ROOT_Folder. It has following folder structured just like Pascal VOC dataset.
```
	<Folder Name> (like progress in example)
	|	Annotations
	|	|*.xml
	|	ImageSets
	|	|	Main
	|	|	|test.txt
	|	|	|train.txt
	|	JPEGImages
	|	|*.jpg
```
3. The default image size saved in new created dataset is 640*480. If you want to specify the size please go through main.py for detail or any other customized options. 
