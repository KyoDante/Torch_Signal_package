### 如何进行数据集管理

1. scenario / person / sample

2. 保存在json或者numpy直接加载？

3. 我们的数据集通常是完整的数据集，
因此需要先分为training set和testing set（可能还有validation set）

4. 然后，在training set部分进行n-way-k-shot的task采样，进行训练。