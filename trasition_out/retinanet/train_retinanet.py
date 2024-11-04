import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, Normalizer
from retinanet import coco_eval

def main():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='../coco/')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=1)
    parser.add_argument('--batchsize', help='Batch size', type=int, default=16)
    args = parser.parse_args()

    # Create the data loaders
    dataset_train = CocoDataset(args.coco_path, set_name='train2014',
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CocoDataset(args.coco_path, set_name='val2014',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = torch.utils.data.RandomSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batchsize, sampler=sampler, collate_fn=collater, num_workers=3)
    dataloader_val = DataLoader(dataset_val, batch_size=1, collate_fn=collater, num_workers=3)

    # Create the model
    if args.depth not in [18, 34, 50, 101, 152]:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    retinanet = getattr(model, f'resnet{args.depth}')(num_classes=dataset_train.num_classes(), pretrained=False)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet)

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    for epoch_num in range(args.epochs):
        retinanet.train()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss
                if loss == 0:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print(f'Epoch: {epoch_num} | Iteration: {iter_num} | Classification loss: {classification_loss:.5f} | '
                      f'Regression loss: {regression_loss:.5f} | Running loss: {np.mean(loss_hist):.5f}')

            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

    torch.save(retinanet.module.state_dict(), f'{args.coco_path}_retinanet_{epoch_num}.pt')

if __name__ == '__main__':
    main()
