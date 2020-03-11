
[keras imdb LSTM demo](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)

[tensorflow ptb LSTM demo](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb)

[pretrained LSTM KWS model](https://github.com/ARM-software/ML-KWS-for-MCU/tree/master/Pretrained_models)

[Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

below code to generate wav file list used to do quantization for kws

```python
import glob, os, json
DATA="D:/tmp/speech_dataset"
L = []
for lbl in ["yes","no","up","down","left","right","on","off","stop","go"]:
  L.extend([os.path.abspath(p) for p in glob.glob("%s/%s/*.wav"%(DATA, lbl))[:10]])
with open('feeds.json', 'w') as f:
  json.dump({'wav_data': L }, f)
```

python tf2lwnn.py -i ML-KWS-for-MCU/Pretrained_models/Basic_LSTM/Basic_LSTM_S.pb -o kws -s wav_data 1,32000 --feeds D:/tmp/kws/feeds.json