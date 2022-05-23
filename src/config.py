common_config = {
    'img_width': 128,
    'img_height': 48,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
    'data_dir': '../data/',
    'checkpoints_dir': '../checkpoints/'
}

train_config = {
    'lr': 0.0005,
    'epochs': 100,
    'train_batch_size': 32,
    'valid_batch_size': 64,
    'show_interval': 100,
    'valid_interval': 500,
    'save_interval': 2000,
    'cpu_workers': 4,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'reload_checkpoint': None,
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 4,
    'reload_checkpoint': 'model.pt',
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
