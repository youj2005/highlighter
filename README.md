# Automatic Highlighter PyTorch Project

### This project uses my own notes to train a LM model to predict which words/tokens will be highlighted and which color/class it will be highlighted.

 The model is built on the **encoder** from *'google/electra-base-generator'* and the **classifciation head** is built using PyTorch. The encoder model is frozen for the first 3 epochs of training to train the classification head and subsequently the encoder is unfrozen to fine-tune the encoder. The data is also resampled to include equal amounts of each highlight color (4 colors)

 *The model achieves **~73% macro-average accuracy** and **~70% micro-average accuracy**.*
 

The model exists in the highlight.py file and all helper methods and objects are wrapped in helper.py