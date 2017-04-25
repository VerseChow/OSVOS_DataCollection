# OSVOS_DataCollection
Data collection via OSVOS method in DAVIS Challenge

The use of this data collection tool is for automatically annotate bounding box in each image for object detection. 

The main idea came from [OSVOS: One-Shot Video Object Segmentation](https://arxiv.org/pdf/1611.05198.pdf)

Input: Vedio Frames, First Frame Ground Truth Segmentation

Output: Segmentation for Each Frame, Bounding Box information

# Pipeline
1. Train with Feed first frame into FCN network to extract the feature of the object wanted to be segmented until loss get down a threshold
2. Feed in the rest vedio frames to get segmentation of the object and Bounding Box information of the object.

