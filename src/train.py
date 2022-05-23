import os
import torch
from torch.nn import CTCLoss
from dataset import VerifyDataset, verify_collate_fn
from torch.utils.data import DataLoader
from config import train_config as config
from evaluate import evaluate
from model import CRNN


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5)
    optimizer.step()
    return loss.item()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 一些参数获取
    lr = config['lr']
    epochs = config['epochs']
    data_dir = config['data_dir']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    train_batch_size = config['train_batch_size']
    valid_batch_size = config['valid_batch_size']
    cpu_workers = config['cpu_workers']
    checkpoints_dir = config['checkpoints_dir']
    reload_checkpoint = config['reload_checkpoint']

    # 训练集和验证集
    train_dataset = VerifyDataset(data_dir, "train")
    valid_dataset = VerifyDataset(data_dir, "valid")
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=cpu_workers, collate_fn=verify_collate_fn)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size,
                              shuffle=True, num_workers=cpu_workers, collate_fn=verify_collate_fn)

    # CRNN
    num_class = len(VerifyDataset.CHARS) + 1
    crnn = CRNN(num_class, map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        print('loading model:',end=' ')
        crnn.load_state_dict(torch.load(
            os.path.join(checkpoints_dir, reload_checkpoint), map_location=device))
        print('over')
    crnn.to(device)

    optimizer = torch.optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to(device)

    # training
    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print(f'train_batch_loss[{i}]:', loss / train_size)

            if i % valid_interval == 0:
                evaluation = evaluate(crnn, valid_loader, criterion,
                                      max_iter=config['valid_max_iter'],
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
                print('valid_evaluation: loss={loss}, acc={acc}'.format(
                    **evaluation))

                if i % save_interval == 0:
                    prefix = 'crnn'
                    loss = evaluation['loss']
                    save_model_path = os.path.join(checkpoints_dir,
                                                   f'{prefix}_{i:06}_loss{loss}.pt')
                    torch.save(crnn.state_dict(), save_model_path)
                    print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    train()
