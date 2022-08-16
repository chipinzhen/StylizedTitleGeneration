# StylizedTitleGeneration
The objective of this project is to design a model to generate 3 different kinds of headlines(Chinese) of news.

The style here are offcial style, normal style, and clickbait style.

The model is based on the BART from huggingface.
The style guided mechanisms are applied in Decoder.



The cleaned version dataset can be collected from https://drive.google.com/drive/folders/1aCBBZAiDwC_Duumr3vRJGF30vtGuAgrB?usp=sharing

The Framework of model
----
![image](https://user-images.githubusercontent.com/24407682/184835369-0169827f-4381-4d66-82f9-2b7a011f5a33.png)
The two tasks share the encoder.

The style guided decoder
----
![image](https://user-images.githubusercontent.com/24407682/184835799-d7ab00b1-178f-4fce-95a7-7341252095fa.png)

As shown in above figure, The style guided mechanisms are implemented by using style-guided normalization layer and style guided attention layer.
