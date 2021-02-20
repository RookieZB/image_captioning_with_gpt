## Description
The idea of this model, is to concatenate image features and text features along the length axis in GPT's attention modules, so the attention weights will be assigned to both image pixels and text tokens. This implementation is mainly for fun, and may contain some mistakes. For running this code, need to download pretrained weights of GPT-2 (124m) and MobileNet V3 (small, dm=1, float) at first.
## References
https://github.com/tensorflow/models  
https://github.com/google-research/bert  
https://github.com/openai/gpt-2