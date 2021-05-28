# OpenCV-and-CNN-Sudoku-Solver-
Project of a Sudoku Solver capable of detecting a Sudoku on an image, and return the input image with the solution overlayed.
<p align="center"><img src="/results/results.jpg" width="900"/></p>


# Table of Contents
   * [What is this?](#What-is-this)
   * [Requirements](#Requirements)
   * [How to use](#How-to-use)
   * [Description](#Description)

# What is this?
This project is based on clear motivation, which is applying a lot of Computer Vision Image Processing learned on the degree, and expand our knowledge to learn about CNN and Deep Learning. We've always liked Sudokus, and the idea to develop a project like this was fascinating to us. This was a perfect excuse to put a lot of hours to a project to apply the best processing of the image possible, made in different depths of the image. We also wanted to learn more about CNNs and how to develop one to do image classification, such as digit classification given an image.

# Requirements
- Python 3.9.x
- Numpy
- Matplotlib
- Pandas
- TensorFlow (Keras)
- OpenCV
- Math
- Sklearn
- Pytesseract

# How to use
1. Clone this repo.
> git clone https://github.com/LeBrav/OpenCV-and-CNN-Sudoku-Solver.git
2. Install the required libraries.
3. Run main.py (selecting images from figs at the start of the .py).

# Description
In this project, given an input of a Sudoku image, we process the image to obtain the Sudoku and it's numbers. We then Solve it using Backtracking, and overlay the solution in the input image. The main work of this project is different techniques used in the processing of the input image, and the creation of a very good CNN model predictor. 
A complete article about the project (written in Catalan), can be found in the main page of this GitHub (name: article (catalan)).
