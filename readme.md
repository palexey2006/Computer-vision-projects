### My computer vision models, that I created and trained by myself using different techniques
### My first step in computer vision: CNN model that trained on CIFAR10 dataset
SOMETHING TO WRITE  
### CIFAR10_CNN model vs CIFAR10_efficient_CNN model
The main difference between them, that the first model uses maxpooling(2x2) to downscale image, while CIFAR10_efficient_CNN uses stride=2 and also a batch_normalization().
The second model is way more efficient, because I spent about 35 minutes to train the first model, while training for the second model took no longer than 5 minutes and gave almost the same results(but if I added a few more epochs in my second model code, I bet it'd perform much better and faster than the first model).Though I added two more conv layers in my second model, it trained much faster.
