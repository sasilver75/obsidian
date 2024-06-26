---
aliases:
  - Open Neural Network Exchange
---


References:
- Video: [Official PyTorch Documentary (At about 15:00 mark)](https://www.youtube.com/watch?v=rgP_LBtaUEc&t=16s)

When PyTorch was being developed at Facebook in 2017/2018, they had finally gotten to the point where the researcher team was being productive with it, but they noticed that there wasn't a lot of transfer between the production ML teams and the research ML teams. This is because the production ML teams were using Caffe2, rather than PyTorch. There was a FB project to build an intermediate language called Taffe that would help somehow bridge PyTorch and Caffe2, to better productionize research code. This project was positively received internally at Facebook, so they decided to open-source the project with Microsoft, AWS, NVIDIA, and other partners -- this project was named ==ONNX (Open Neural Network Exchange)==. 

A year or two later, it seemed like with a lot of consolidation in the deep learning framework space, there were really only two options: PyTorch and Tensorflow. So what was the point of ONNX anymore? So they started the PyTorch 1.0 effort, trying to combine the frontend, intermediate sections, and backend, to have a unified stack.
- The awesome frontend of PyTorch that's easy to use, and the high-performance backend of Caffe2... and they tried to zip them together, but it was too hard.
- So they tried to keep the PyTorch codebase and just optimized it for production as hard as they could. This took a good two years to evolve PyTorch into something that could be productionized -- PyTorch 1.0.

Libraries exploded on PyTorch; the whole world started to build on it! A bunch of self-driving car companies (Tesla, Cruise, Uber) started using PyTorch for self-driving. Soon, even Google had enough pressure/interest to run PyTorch on their own TPU hardware for customers!