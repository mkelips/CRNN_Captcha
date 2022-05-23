import torch
import numpy as np
from PIL import Image
from dataset import VerifyDataset

from model import CRNN
from decoder import decode
from config import common_config as config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

path = "../image/0_mh3I.jpg"
image = Image.open(path).convert('L')
image = image.crop((40, 2, 168, 50))
image = np.array(image)
image = image.reshape((1, 48, 128))
image = (image / 127.5) - 1.0
image = torch.FloatTensor(image)
image = image.unsqueeze(1)
image = image.to(device)

print('loading model:', end=' ')
reload_checkpoint = '../checkpoints/model.pt'
num_class = len(VerifyDataset.CHARS) + 1
crnn = CRNN(num_class, map_to_seq_hidden=config['map_to_seq_hidden'],
            rnn_hidden=config['rnn_hidden'],
            leaky_relu=config['leaky_relu'])
crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
print('over')

with torch.no_grad():
    logits = crnn(image)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    preds = decode(log_probs, method='beam_search', beam_size=10,
                   label2char=VerifyDataset.LABEL2CHAR)

    pred = "".join(preds[0])
    print(pred)
