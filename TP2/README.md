# Deep Learning Project

This is the README for our Deep Learning TP on the context of a project in Image Mining,
during our Master M2 studies in AI, Univeristé Paris-Saclay.

This README itself and the code used to produce the presented results are held freely on

[TP2 repo](https://github.com/gpspelle/image-mining/tree/master/TP2)

Go check it! 

## Crew members

[@gpspelle](https://github.com/gpspelle)
[@damounayman](https://github.com/damounayman)

## Database

The well-known CIFAR10 database will be used on our experiments.

## Environment

A google colab file was provided by the professor and our goal is to fill it and make
some analyzes about the results we got and also fine tune our hyper parameters.

## Tasks to be accomplished on this README

This readme explain in fair words the work done and present some results. Also,
one may check the code directly instead of reading this file. However, on the code
only the best set of parameters are available and here a deeper analysis on the
used process is described.

### Detailing Part III

TODO: Explain what neural networks and hyper parameters configurations you tested, the resulting performances and trade-offs found.

So, after executing the provided notebook once, my first approach was to try a VGG16 network, since I have already used it before
and it sounded a great idea to check its results on this dataset. However, a full VGG16 looks like too much for these small images.
CIFAR is composed of 32x32 images, too small to unleash a VGG16 over it. Thus, I cropped this network and just used the first block,
and even the first block was used with different parameters.

Some trade-offs that need to be analyzed are the size of the batch used, because the smaller the batch, faster is the training process.
Because with a great batch, you'll use the result of a lot of images to update your net.

If you use a lot of conv layers with a high number of features, you'll probably use a cannon too big for this problem. Thus, it's necessary
to balance the number of filters of your conv layers and also the size of your fully connected layers in the end.

The number of epochs is more or less experimental, I changed from 10 to 4 on the first experiment, because I saw that the net was overfitting after
a few epochs. The learning rate is more or less experimental and you need to have the feeling between your epochs if it's moving in the training
phase or if it's stagnated.

### Loss interpretation

[Configuration number 2 loss plot](figures/number_2_loss.png)


As seen on the figure above, the loss on validation is starting to converge and even increase close to the value of 1.0. But, the training loss
is going doing and down. This pattern: training loss going down and validation loss going up is the signal that you can stop training and close
your model, it's already overfitting to the training data.

Something that's interesting and possible to see on figures below, is the effect of the dropout on the last connected layers.

[Configuration number 3 loss without dropout](figures/number_3_loss_without_dropout.png)
[Configuration number 3 loss with dropout](figures/number_3_loss_with_dropout.png)

Note that the difference between the figures above also take in consideration an epoch number increase, with dropout, the problem
is harder to learn, but more general. Since, a dropout of 0.65 is being used, 65% of the connected layers are erased during training.
So, more epochs are given to the network. 

One improvement that can be used is to lower the learning rate and increase the number of epochs, but that costs a lot in
computational time necessary. It's still possible to increase the number of convolutional layers being used, but I believe
that these set of fully connected layers are already ok for this problem. Maybe increase a little the size of it.

### Best set of parameters

Our very best results were obtained with the following set of parameters:

|         Parameters  | #1      | #2       | #3         | #4         |
| :-----------------: | :-----: | :------: | :--------: | :--------: |
|      Learning rate  | 0.001   | 0.001    | 0.0001     | 0.0001     |
|         Batch size  | 32      | 256      | 32         | 32         |
|       N. of epochs  | 10      | 4        | 4          | 10         |
|      Test accuracy  | 63.57%  | 67.93%   | 69.20%     | 
|      Training size  | 80%     | 80%      | 80%        | 80%        |
| N. of feature maps  | 64, 128 | 128, 128 | 64, 128    | 64, 128    |
| N. of conv. layers  | 2       | 2        | 2          | 2          |
| N. of conn. layers  | 1       | 1        | 2          | 2          |
|     FC layers size  | 1024    | 4096     | 1024, 1024 | 1024, 1024 |
| Dropout             | No      | No       | No         | Yes        |

Many sets are detailed, since the way we change one hyper-paremeter might change the way we need to change another.
One cool way to solve this tuning of parameters problem is to use AutoML with them.
