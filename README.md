### This Capsulation Library consists of Deep Learning with PyTorch and some signal processing with scipy(or others).

---

##### Library structure
- data
    - /dataset/ is where you store your data
    - /my*.py provides an example realization of torch.dataloader

- models
    - /few-shot/ provides some networks about few-shot learning
    - /machine-learning/ provides some machine learning algorithms, including classification, regression and clusters.
    - /*.py provides some classical network architechtures.

- signal-processing
    - /features/ provides time、frequency、energy domain features extraction.
    - /filters/ provides some signal filters.
    - /transformation/ provides some signal transformation (time->freq or freq->time, etc)

- utils
    - /pytorch2mobile provides some converter on how to convert PyTorch model to ncnn or TorchMobile
    - /*.py provides some tools to handle time calculation, etc.

- visualization
    - /*.py provides some visualization functions.
