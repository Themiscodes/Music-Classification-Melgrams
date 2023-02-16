# Music Classification using Melgrams

This repository contains an implementation of a music genre recognition system using Convolutional Neural Networks (CNN). Two different approaches were used for feature extraction: Mel-frequency cepstrums (MFCC) and Mel Spectrograms.

This project aims to provide a comprehensive understanding of the fundamental principles behind developing deep neural networks for music genre recognition. The code along with the detailed analysis can be found in the [music_genre_classification.ipynb](music_genre_classification.ipynb) notebook.


## Classification

Music genre classification is a popular application of machine learning and one approach to this problem is to use Convolutional Neural Networks with Mel spectrograms as input. Mel spectrograms are a type of visual representation of audio signals that are commonly used in audio analysis and music information retrieval and they use a frequency scale that is based on the mel scale.

![mel](melgrams.png)

The model achieved over 80% accuracy and F1-Macro averaged score on the test set, but it had poor performance on new data. To combat this, several regularization techniques were implemented, including batch normalisation, dropout, weight decay and early stopping. Dropout and weight decay were used to reduce the complexity of the model and prevent it from fitting noise in the training data, while early stopping was used to prevent the model from overfitting by stopping the training process before the model started to perform poorly.

To further improve the model, more sophisticated techniques were employed, including max pooling and padding for down-sampling and preserving the spatial dimensions of the data, respectively. Finally, hyperparameter tuning was conducted by experimenting with different learning rates, batch sizes and activation functions and a variety of learning rate schedulers and optimizers were tested to find the optimal configuration.

## Music Recognition

When the model was applied to classify music downloaded from YouTube and converted to Mel spectrograms, it performed well in most cases, which suggests that the regularization techniques were effective in improving the model's generalization performance.

### Blues

![blues](sonhouse.png)

- Regarding the "Blues" genre, the model's predictions were mainly concentrated on sections of the songs where the guitar and high frequencies were prevalent. However, during parts where other instruments like the bass were more dominant, the model classified them as either "Hip Hop" or "Rock". When classifying older songs though, like Son House's above, the model correctly classified the majority of it. This is likely because the song has minimal post-production additions and consists of just vocals and guitar.

### Classical

![classical](mozart.png)

- The model achieved even better accuracy when predicting classical music, but it still misclassified sections of the songs as a different genre. When listening to these timestamps, it is evident that there are changes in the tone of the song that the model may be picking up on. For instance, Mozart's Lacrimosa is predicted as classical music with 80% accuracy, but there are seconds where the model classifies it as "Blues", which coincide to sections of the piece where the orchestra is not singing. 

### Hip Hop / Pop

![nas](nas.png)

- The model had high accuracy when classifying Hip Hop songs with the "Hip Hop/Pop" genre, but struggled with some Pop songs. For instance, when classifying Nas' song, the model accurately predicted 90% of it as "Hip Hop". This is expected of course as such songs usually have a consistent, repetitive beat without many fluctuations.

### Rock

![rock](nirvana.png)

- When classifying Rock songs, the model encountered the most difficulty. For instance, in the song "Drain You" by Nirvana, which starts with guitar and vocals, it is classified as Blues and Classical until the drums start in the 8th second, which is classified as Rock. It seems that the drums have some melodic feature that the model has learned for this classification. This trend was also observed in other Rock songs.

In conclusion, the model's genre predictions were quite accurate for most of the duration of the songs. The Convolutional Neural Network performed better in predicting the genre of older songs with consistent melody and frequency patterns. There were many instances where the genre was misclassified, but these outliers could also be attributed to the similarities in the frequency spectra of modern music recordings across genres. 
