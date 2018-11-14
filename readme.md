# 图片分类模版

	1. 这是一个用于图像分类的模版，使用此模版可以快速利用gluon/pytorch中的model_zoo 中的模型进行训练，
	从而得到一个基础的baseline。
	2. gluon_model.py 用于gluon模型的训练
	3. torch_model.py 用于pytorch模型的训练
	4. process.py 用于图像处理。
	5. utils.py 中定义了用于分类的一些函数，用来辅助训练和描述结果

# utils.py

	1. 定义了一些可用于gluon的图像增强方法，eg. RandomRotate (pytorch 的图像增强方法比较全)
	2. lr_find 用于辅助判断gluon模型的初始学习率
	3. get_logger用于方便记录log
	4. ConfusionMatrix 用于计算分类结果的混淆矩阵
	5. TrainerWithDifferentLR 用于gluon对不同参数设置不同的初始学习率，使用时可根据具体模型修改。
	6. 其他具体见代码


	