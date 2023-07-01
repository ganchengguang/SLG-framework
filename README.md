# SLG-framework
paper code
https://arxiv.org/abs/2306.15978

Sentence-to-Label Generation Framework for Multi-task Learning of Japanese Sentence Classification and Named Entity Recognition


You need change Transformer code with our to use Constraint Mechanism. 
please must use transformers 4.19.2 version. And replace the modeling_transformer_t5 three py file of origin transformers's code.


And Incremental Learning is use no predict setep fine-tune code.

Requiment  
CUDA 11.6
torch              1.13.1+cu116  
torchaudio         0.13.1+cu116  
torchvision        0.14.1+cu116  
transformers       4.19.2  
pandas             1.4.4  
scikit-learn       1.1.2  


