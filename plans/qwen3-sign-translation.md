## Train a multimodal qwen model for sign language translation

### Instruction:
1. Initialize the model from scratch using `Qwen3-0.6B` for text decoder and `GestureEncoder` for multimodal encoder, which was in `/mnt/data4/outputs/gesture_pretraining_contrastive_150m`, pretrained using `/home/slimelab/Projects/slt/gesture_pretrain.sh`.
2. The output of the gesture encoder should be concatenated with text embeddings of the prompt and fed to the text decoder. The decoder will generate the translation text.
3. Use the **whole dataset** in `/mnt/data2/sign_gestures` for training. Select only the train partition. Set max frames to 512, if longer than 512, downsample to 512 rather than truncation. 
4. Also prepare val and test code. For evaluation, compute all metrics and save the predicted texts. You can refer to `/home/slimelab/Projects/slt/src/t5_slt`. 
5. Use the same prompt format as `/home/slimelab/Projects/slt/src/t5_slt` script. You can reuse the dataloader for the iterative dataset for loading webdataset. 
6. Keep training in full bf16 for both gesture encoder and text decoder.
7. Allow multiple GPUs for training.
8. Allow multiple training settings. 
    - unfreeze the gesture encoder before the projector
    - unfreeze the projector of the gesture encoder
    - unfreeze the text decoder
9. For tokenization and preprocessing, you can refer to how qwen3-vl does it from the `transformers` source code. 
10. Continue to use the `slt_qwen` conda environment for training. Apply all available optimizations like liger kernel and flash attention 2. there are also cross entropy kernels for causal modeling. 
11. Start a new folder under `/home/slimelab/Projects/slt/src` for the code. 
