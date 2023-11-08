# anEye - A real time AI tool to assist navigation for visually impaired people
This is a prototype for my thesis proposal at Daffodil International University.

## Abstract
Navigating from one place to another has been a problematic task for the blind. In Bangladesh, the existing footpaths are mostly crowded or broken. Often, visually impaired people get hurt while walking in a footpath as they do not have anything but a stick to help them. Considering the problem scenario, we are proposing a smart solution to identify safe footpath and detect obstacles in a footpath. The system will also be capable of estimating the distance of the object as well as suggesting the safe pathway. To train the models we built a dataset of footpath images of Dhaka containing 3,000 hand-annotated RGB images for semantic segmentation and another dataset containing 500+ samples of real-world distance of reference objects w.r.t to their pixel coordinates in an image for distance estimation. We proposed an optimized U-Net architecture that is trained on our segmentation dataset which is capable of inference safe footpath with 96% accuracy with as low as 4.7 million parameters. The system utilizes YOLOv3 architecture for object detection and a polynomial regression based novel approach to estimate object distance. The distance measurement model obtains 94% score.

## Dataset Link
Kaggle: https://www.kaggle.com/datasets/afifaniks/footpath-image-dataset
