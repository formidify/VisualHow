# VisualHow: Multimodal Problem Solving

This code implements the VisualHow dataset with four different tasks:

- Solution Steps Prediction
- Solution Graph Prediction
- Problem Description Generation
- Solution Captions Generation

Disclaimer
------------------
For the Solution Steps Prediction, Solution Graph Prediction and Solution Captions Generation, we adopt the repository [vse-infty](https://github.com/woodfrog/vse_infty)[1], while for Problem Description Generation, we adopt the repository [AREL](https://github.com/eric-xw/AREL)[2] and [UpDn](https://github.com/nocaps-org/updown-baseline)[3].

Requirements
------------------

- Python 3.7
- PyTorch 1.2.0 (along with torchvision)
- Transformers 4.8.2 

Reference
------------------

[1] Jiacheng Chen, Hexiang Hu, Hao Wu, Yuning Jiang, and ChanghuWang. Learning the best pooling strategy for visual semantic embedding. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021.

[2] Xin Wang, Wenhu Chen, Yuan-Fang Wang, and William Yang Wang. No metrics are perfect: Adversarial reward learning for visual storytelling. In *Annual Conference of the Association for Computational Linguistics (ACL)*, 2018.

[3] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In Proceedings of the *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.
