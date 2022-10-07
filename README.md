# OpenCV-Camera-Calibration-with-Vanishing-Points

This program estimates the camera calibration matrix K from three vanishing points in an image.

## The original Image
![image](https://user-images.githubusercontent.com/69100847/194550614-d3feaebe-f14b-4d26-b894-9075ed7dddae.png)


## Visualization of vanishing points
![image](https://user-images.githubusercontent.com/69100847/194550731-22b3057c-c010-4e03-b159-ebec01320ab3.png)


## Estimating the coordinates of vanishing points through least square
![image](https://user-images.githubusercontent.com/69100847/194550827-81cb5adb-3feb-4ee8-be39-035b61950cbb.png)

## Solve K matrix Using the location of vanishing points
### We make use of the orthonormal constraints of the rotation matrix to remove the effect of scaling factor

![image](https://user-images.githubusercontent.com/69100847/194551002-d6c70d72-2652-48fd-b943-6d87b11d92b9.png)
