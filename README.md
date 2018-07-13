[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/select_kernel.png "select aind-vui kernel"

## Project Overview

This project was submitted as final project for second semester of Artificial Intelligence Nanodegree at Udacity that focused on Deep Artificial Neural Networks
In this project a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline was built!
The theory is based on following textbooks and articles
[Source and Filter speech models, Forier Analysis and MFCC](http://web.science.mq.edu.au/~cassidy/comp449/html/ch07.html#d0e1094)


[Deep speech 2: End-to-end speech recognition in english and mandarin. International Conference on Machine Learning. 2016.](https://arxiv.org/pdf/1512.02595v1.pdf )

![ASR Pipeline][image1]

## Project Results

I used data from libriSpeech and TedLium datasets for obtaining trained speech data
![ScreenShot](/screenshot/graph1.png)


Note on graphs:

Top row graphs are for models 0 - 4 basic models trained on the small dataset. Bottom row graphs are for more advanced models trained on larger dataset. All the models here are trained on a combined 360 +100 hours dataset, except model 5 which is trained on 360 hour dataset only.

Basic Models

Model 0: - Simple 1 layer rnn model. Two variants were trained one using the spectogram as input and another mfcc features

Model 1: - Two layer model that adds time distributed layer on top of Rnn layer

Model 2: - The cnn layer is added before model 1 to extract features from spectogram input

    baseline - SimpleRnn cells
    gru1 - standard gru cell activation
    gru - using 'relu' instead of 'tanh' activation
    lstm - lstm cell 

Model 3: - Instead of adding cnn several Rnn layers are used

Model 4: - Bidirectional Rnn

Experimental Models

model 5: is similar to model2 lstm, but trained on larger 360 hour dataset

model 6_3: 3 layer gru rnn trained on 460 hours of transcribed speech

model end: 1 layer cnn + 4 layer gru rnn with 0 dropout

model_end3: 1 layer cnn + 3 layer gru rnn with .2 dropout

model_end42: 1 layer cnn + 4 layer gru rnn with .2 dropout

Results:

Basic models

model 0:

It is clear that the default model0 that uses just one RNN layer is too simple to deal with the data complexity and both its training and validation loss are very high.

model1:

Addition of the Time distributed layer on top of RNN improves the results drastically. For the amount of data that is tested (~2000 labelled examples) it reaches similar results to more complex models, without much overfitting. So it might be chosen if the amount of data does not increase.

model2:

The addition of CNN layer to extract features from the spectogram data clearly improves the model capacity, however at around 8 epochs the validation loss reaches it's minimum and further training leads to decreased performance and overfitting. The variant that uses non-standard gru cell with 'relu' activation outperforms other models of this type slightly. While standard GRU and LSTM perform about the same and slightly better than SimpleRnn

model3:

Increasing the depth of the model by adding several rnn layers also has positive effect on the model capcity and its validation loss. The model with 2 gru layers performs better than model with 1 and model with 3 gru layers performs best.

model4:

Bidirectional layer is performing better than single gru rnn layer, however it seems that stacking two regular rnn layers on top of each other outperforms a bidirectional layer so regular stacking works better for the same number of parameters

Experimental models:

model5:

This model is exactly the same as model 2 lstm. The only difference is that it is trained on more data and reduces validation loss by about 50%

model6_3:

This model is trained on even larger dataset than model 5 and has exactly the same architecture as model_3_3 with 3 rnn layers stacked on top of each other. Clearly training on more data has potential for improving validation loss.

model_end:

This is a variant of the final architecture 1 Cnn layer for extracting features before a number of rnn layers. This variant has 4 rnn layers and 0 dropout. It produces the lowest validation loss of all the models tested, along with model_end42. However unlike model_end42 it is less stable and can suffer from overfitting when overtrained. It reached its lowest validation loss at epoch 24 and then the results started to decrease again.

model_end3:

This is a variant of 1 cnn layer 3 rnn layers with .2 dropout. It did not show as much promise as model_end42, so it was run only for 20 epochs.

model_end42:

This is the best model. By adding .2 dropout to model_end, the training process is stabilized. The training loss is much higher with dropout present, than in model withouth dropout, but it practically stops decresing indicating that no overfitting occurs. The model reaches what seems to be very close to the optimal performance at epoch 40. The final loss of 64.0 is very. It might be possible to decrease loss by a couple of points by further training, but judging from the graph the model seems to be very close to its optimum performance. (See more in answer to question 2)


## Final Architecture

For the final architecture I decided to choose the architecture with CNN layer that would preprocess spectrogram input and prepare the features for the input into several layers of RNN that act similar to phonetic and language models. The preliminary results indicated that simply stacking the RNN layers on top of each other was slightly more effective than the bidirectional RNN, so that is the architecture I chose. I experimented with the amount of RNN layers and dropout and finally the architecture with 1 CNN layer and 4 rnn layers and .2 dropout seemed to produce the best and most stable results. One final note. I used the combined 100 hour, 360 hour dataset, 500 hour dataset and TEDLIUM3 dataset for training the model. The clear trend from the experiments was that the more the amount of data I have for training the better the results are irrespective of the exact architecture used. Also with the increased amount of data the bigger models tended to become more powerful and overfit less.

Future work: I also worked on a separate language model for which I downloaded 3gb worth of text files from project gutenberg. I preprocessed the files by removing them from archives and removing irrelevant header and footer texts, and then removing all the characters except letters and space and converting to lowercase. Due to the time constraints I could not finish model training or incorporate the language model into ASR. I hope to do it in the near future



