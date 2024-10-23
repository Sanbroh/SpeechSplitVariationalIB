# Unsupervised Speech Decomposition Via Triple Information Bottleneck

This repository provides a PyTorch implementation of [SpeechSplit](https://arxiv.org/abs/2004.11284), which enables more detailed speaking style conversion by disentangling speech into content, timbre, rhythm and pitch.

This is a short video that explains the main concepts of our work. If you find this work useful and use it in your research, please consider citing our paper.

[![SpeechSplit](./assets/cover.png)](https://youtu.be/sIlQ3GcslD8)

```
@article{qian2020unsupervised,
  title={Unsupervised speech decomposition via triple information bottleneck},
  author={Qian, Kaizhi and Zhang, Yang and Chang, Shiyu and Cox, David and Hasegawa-Johnson, Mark},
  journal={arXiv preprint arXiv:2004.11284},
  year={2020}
}
```


## Audio Demo

The audio demo for SpeechSplit can be found [here](https://auspicious3000.github.io/SpeechSplit-Demo/)

## Dependencies
- Python 3.6
- Numpy
- Scipy
- PyTorch >= v1.2.0
- librosa
- pysptk
- soundfile
- matplotlib
- wavenet_vocoder ```pip install wavenet_vocoder==0.1.1```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder


## To Run Demo

Download [pre-trained models](https://ibm.box.com/s/kgomuly35meo8xsh5mfxxklol8bulrry) to ```assets```

Download the same WaveNet vocoder model as in [AutoVC](https://github.com/auspicious3000/autovc) to ```assets```

The fast and high-quality hifi-gan v1 (https://github.com/jik876/hifi-gan) pre-trained model is now available [here.](https://ibm.box.com/s/asvv554v0zd09yipl2qadz49i7jpdhta)

Run ```demo.ipynb``` 

Please refer to [AutoVC](https://github.com/auspicious3000/autovc) if you have any problems with the vocoder part, because they share the same vocoder scripts.


## To Train

Download [training data](https://ibm.box.com/s/ahaj5zbuwu7jox47zxnsls2syf12g4c5) to ```assets```.
The provided training data is very small for code verification purpose only.
Please use the scripts to prepare your own data for training.

1. Extract spectrogram and f0: ```python make_spect_f0.py```

2. Generate training metadata: ```python make_metadata.py ```

3. Run the training scripts: ```python main.py```

Please refer to Appendix B.4 for training guidance.


## Final Words

This project is part of an ongoing research. We hope this repo is useful for your research. If you need any help or have any suggestions on improving the framework, please raise an issue and we will do our best to get back to you as soon as possible.


