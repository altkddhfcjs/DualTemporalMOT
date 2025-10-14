_base_ = ['base.py']

train_dataset = {'MOT20':['train'], 'Crowdhuman':['train', 'val']}

test_dataset = {'MOT20':['train']}

num_classes = 91