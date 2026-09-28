# <span style="color:green">Rubik's cube solver</span>.

This project implements a Rubik's cube solver using the Kociemba algorithm for suboptimal solutions. For convinience, a web interface is implemented to provide the users with the opportunity to either manually enter the cube sides or use the color recognition functionality by uploading an image of the cube side.

## The Dataset

Due to the lack of an appropriate labaled dataset available a custom one was created. The dataset size is relatively small - around 12 000 examples, but the models evaluate good on it. Each example is a 20x20 pixel image colored in one of the cube colors (white, red, blue, green, yellow, orange).

## Cube recognition

The approach used to make predictions for each side of the cube is the following:
- Detect the edges of the cube face and interpolate the centers of each cubie
- Make a prediction for each cubie color

### Edge detection and cubie center interpolation

Canny edge detection is used to find the edges of the uploaded image. Then the leftmost, rightmost, upmost and downmost points of the edges are used to find the vertices of the face. After that using simple geometry the centers of the cubies are found. 
(Experiments were held with Convolutional Neural Networks for the corner detection, but did not yeld great results, probably due to the necesity of more training images, despite data augmentation) 

<img title="Edge detection" alt="Edge detection example" src="/project_images/edge_detection.png">

### Color predictions

For the color recognition several simple scikit learn models were tested and compared to other using Convolutional Neural Networks (implemented using pytorch). You can find the full model report in **model_report_color_prediction.xlsx**. The results show that the simple KNN with 3 neighbors performed slightly better than the second best CNN model. Those two best models are sved in the **color_classification_models** folder. In the solving server there is the option to switch between them.

## The web interface

The web interface is implemented using the [p5.js](https://p5js.org/) library. It gives a 3D view of the modeled cube and a 2D cube unfold where cubie colrs can be changed manually.

<img title="Web interface" alt="Web interface" src="/project_images/interface.png">

The "Solve" button requests the solving steps from the server and displays them. For the move the following notation is used:
- U: rotate the up side (whith white center) clockwise
- F: rotate the front side (whith green center) clockwise
- B: rotate the back side (whith blue center) clockwise
- L: rotate the up side (whith orange center) clockwise
- R: rotate the up side (whith red center) clockwise
- D: rotate the up side (whith yellow center) clockwise
- U': rotate the up side (whith white center) counterclockwise
- F': rotate the front side (whith green center) counterclockwise
- B': rotate the back side (whith blue center) counterclockwise
- L': rotate the up side (whith orange center) counterclockwise
- R': rotate the up side (whith red center) counterclockwise
- D': rotate the up side (whith yellow center) counterclockwise
- U2: rotate the up side (whith white center) clockwise twice
- F2: rotate the front side (whith green center) clockwise twice
- B2: rotate the back side (whith blue center) clockwise twice
- L2: rotate the up side (whith orange center) clockwise twice
- R2: rotate the up side (whith red center) clockwise twice
- D2: rotate the up side (whith yellow center) clockwise twice

IMPRTANT NOTE: 
**<ins> All solving operation should be executed from the perspective of the rotated side!</ins>**
=======
