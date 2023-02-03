import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchtext.vocab import GloVe
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import random_split
import time
import warnings
# from torch.nn.utils.rnn import pad_sequence

from image_datasets import *
from train_helpers import set_bn_eval, CSMRLoss
from model_loader import setup_explainer

warnings.filterwarnings("ignore") # ignore dataloader warnings

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, embeddings, output_path, experiment_name, train_label_idx, k=5):
    print(f'Epoch: {epoch}')
    train_loss_epoch = 0.0
    batch_index = 0
    num_batch = len(train_loader)
    correct = 0.0
    top_k_correct = 0.0
    model.train()
    model.apply(set_bn_eval)
    for _, batch in enumerate(train_loader):
        batch_index += 1
        # print(f'batch idx: {batch_index}')
        # data, target, mask = torch.stack(batch[0], dim=1).cuda(), torch.stack(batch[1], dim=1).squeeze(0).cuda(), torch.stack(batch[2], dim=1).squeeze(0).cuda()
        data, targets, masks = batch[0].cuda(), batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
        predict = data.clone()
        for name, module in model._modules.items():
            if name=='fc':
                predict = torch.flatten(predict, 1)
            predict = module(predict)
            if name==args.layer:
                # if torch.sum(mask) > 0:
                # print(f'predict shape: {predict.shape}, mask shape: {masks.shape}')
                predict = predict * masks
                # else:   
                    # continue
        loss = loss_fn(predict, targets, embeddings, train_label_idx)
        if args.refer=='vg':
            loss /= predict.shape[0]    # normalize by number of masks/concepts
        
        if np.isnan(loss.item()):
            continue
        
        sorted_predict = torch.argsort(torch.mm(predict, embeddings) /
                                       torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                                torch.sqrt(torch.sum(embeddings ** 2,
                                                                     dim=0, keepdim=True))),
                                       dim=1, descending=True)[:, :k]
        for i, pred in enumerate(sorted_predict):
            correct += targets[i, pred[0]].detach().item()
            top_k_correct += (torch.sum(targets[i, pred]) > 0).detach().item()

        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        if batch_index % args.print_every == 0:
            print(f'batch idx {batch_index} | loss: {loss.cpu().item():.3f} | train: [{batch_index}/{num_batch} ({100. * batch_index / num_batch:.2f}%)]')

        if batch_index % args.save_every == 0:
            ckpt_file = f'{output_path}/ckpt_tmp.pth.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt_file)
            


        train_loss_epoch += loss.data.detach().item()
        torch.cuda.empty_cache()

    train_loss_epoch /= (len(train_loader) * train_loader.batch_size)
    print(f'correct: {correct}, top_k_correct: {top_k_correct}')
    print(f'len(train_loader): {len(train_loader)}, batch_size: {train_loader.batch_size}')
    train_acc = correct / (len(train_loader) * train_loader.batch_size * sorted_predict.shape[0])
    train_top_k_acc = top_k_correct / (len(train_loader) * train_loader.batch_size * sorted_predict.shape[0] * k)
    print("\ntrain average loss: {:.6f}\t".format(train_loss_epoch))
    print("train top-1 accuracy: {:.2f}%".format(train_acc * 100))
    print("train top-5 accuracy: {:.2f}%".format(train_top_k_acc * 100))
    
    return train_loss_epoch, train_acc


def validate(model, loss_fn, valid_loader, embeddings, train_label_idx, k=5):
    model.eval()
    valid_loss_epoch = 0
    correct = 0.0
    top_k_correct = 0.0
    for _, batch in enumerate(valid_loader):
        with torch.no_grad():
            # data, target, mask = torch.stack(batch[0]).cuda(), torch.stack(batch[1]).squeeze(0).cuda(), torch.stack(batch[2]).squeeze(0).cuda()
            data, targets, masks = batch[0].cuda(), batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
            predict = data.clone()
            for name, module in model._modules.items():
                if name=='classifier' or name=='fc':
                    if args.model == 'mobilenet':
                        predict = torch.mean(predict, dim=[2, 3])
                    else:
                        predict = torch.flatten(predict, 1)
                predict = module(predict)
                if name==args.layer:
                    predict = predict * masks
            sorted_predict = torch.argsort(torch.mm(predict, embeddings) /
                                           torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                                    torch.sqrt(torch.sum(embeddings ** 2,
                                                                         dim=0, keepdim=True))),
                                           dim=1, descending=True)[:, :k]
            for i, pred in enumerate(sorted_predict):
                correct += targets[i, pred[0]].detach().item()
                top_k_correct += (torch.sum(targets[i, pred]) > 0).detach().item() 

            loss = loss_fn(predict, targets, embeddings, train_label_idx).data.detach().item()
            if args.refer=='vg':
                loss /= predict.shape[0]
            if np.isnan(loss):
                continue
            valid_loss_epoch += loss
            
        torch.cuda.empty_cache()

    valid_loss_epoch /= len(valid_loader) * valid_loader.batch_size
    valid_acc = correct / (len(valid_loader) * valid_loader.batch_size * sorted_predict.shape[0])
    valid_top_k_acc = top_k_correct / (len(valid_loader) * valid_loader.batch_size * sorted_predict.shape[0] * k)
    print('vtlid average loss: {:.6f}\t'.format(valid_loss_epoch))
    print("vtlid top-1 accuracy: {:.2f}%".format(valid_acc * 100))
    print("vtlid top-5 accuracy: {:.2f}%".format(valid_top_k_acc * 100))
    
    return valid_loss_epoch, valid_acc


def main(args, train_rate=0.9):
    word_embedding = GloVe(name='6B', dim=args.word_embedding_dim)
    torch.cuda.empty_cache()

    model = setup_explainer(args, random_feature=args.random)
    if args.model == 'densenet121':
        parameters = model.classifier.parameters()
    else:
        parameters = model.fc.parameters()

    model = model.cuda()
    if not args.name:
        args.name = f'{args.refer}_{args.model}_{args.layer}_ar={args.anno_rate}'
    if args.random:
        args.name += '_random'

    args.save_dir = args.save_dir + '/' + args.name + '/'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    mask_dims = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'features':7, 'denselayer16':7}
    if args.refer == 'vg':
        dataset = VisualGenome(root_dir='data', transform=data_transforms['val'], mask_dim=mask_dims[args.layer])
        datasets = {}
        train_size = int(train_rate * len(dataset))
        test_size = len(dataset) - train_size
        torch.manual_seed(0)
        datasets['train'], datasets['val'] = random_split(dataset, [train_size, test_size])
        label_index_file = os.path.join(args.data_dir, "vg/vg_labels.pkl")
        with open(label_index_file, 'rb') as f:
            labels = pickle.load(f)
        label_index = []
        for label in labels:
            label_index.append(word_embedding.stoi[label])
        np.random.seed(0)
        train_label_index = np.random.choice(range(len(label_index)), int(len(label_index) * args.anno_rate), replace=False)
        word_embeddings_vec = word_embedding.vectors[label_index].T.cuda()
    elif args.refer == 'coco':
        # if you wanna take a subset of data
        #datasets = {'val': MyCocoSegmentation(root='./data/coco/val2017',
        #                                      annFile='./data/coco/annotations/instances_val2017.json',
        #                                      transform=data_transforms['val'],
        #                                      subset=250),
        #            'train': MyCocoSegmentation(root='./data/coco/train2017',
        #                                        annFile='./data/coco/annotations/instances_train2017.json',
        #                                        transform=data_transforms['train'],
        #                                        subset=5000)}
        datasets = {'val': MyCocoSegmentation(root='./data/coco/val2017',
                                              annFile='./data/coco/annotations/instances_val2017.json',
                                              transform=data_transforms['val']),
                    'train': MyCocoSegmentation(root='./data/coco/train2017',
                                                annFile='./data/coco/annotations/instances_train2017.json',
                                                transform=data_transforms['train'])}
        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        label_embedding = torch.load(label_embedding_file)
        label_index = list(label_embedding['itos'].keys())
        train_label_index = None
        word_embeddings_vec = word_embedding.vectors[label_index].T.cuda()
    else:
        raise NotImplementedError
    
    # def collate_fn(batch):
    #     # 
    #     # if args.refer=='coco':
    #         # return tuple(zip(*batch))
    #         # return batch
    #     # elif args.refer=='vg':
    #     imgs, targets, masks = zip(*batch)

    #     # Pad the targets and masks with zeros
    #     targets = pad_sequence(targets, batch_first=True)
    #     masks = pad_sequence(masks, batch_first=True)

    #     # Stack the images together
    #     imgs = torch.stack(imgs)

    #     return imgs, targets, masks

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    loss_fn = CSMRLoss(margin=args.margin)

    # Train and validate
    best_valid_loss = 99999999.
    train_accuracies = []
    valid_accuracies = []
    best_epoch = 0
    with open(f'{args.save_dir}/logs.txt', 'w') as f:
        start_training = time.time()
        for epoch in range(args.epochs):
            start_epoch = time.time()
            train_loss, train_acc = train_one_epoch(epoch, model, loss_fn, optimizer, dataloaders['train'],
                                                    word_embeddings_vec, args.save_dir, args.name, train_label_index)
            ave_valid_loss, valid_acc = validate(model, loss_fn, dataloaders['val'],
                                                 word_embeddings_vec, train_label_index)
            train_accuracies.append(train_acc)
            valid_accuracies.append(valid_acc)
            scheduler.step(ave_valid_loss)
            f.write('epoch: %d\n' % epoch)
            f.write('train loss: %f\n' % train_loss)
            f.write('train accuracy: %f\n' % train_acc)
            f.write('validation loss: %f\n' % ave_valid_loss)
            f.write('validation accuracy: %f\n' % valid_acc)

            if ave_valid_loss < best_valid_loss:
                best_valid_loss = ave_valid_loss
                best_epoch = epoch
                print('==> new checkpoint saved')
                f.write('==> new checkpoint saved\n')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, f'{args.save_dir}/best_model.pth.tar')
                plt.figure()
                plt.plot(train_loss, '-o', label='train')
                plt.plot(ave_valid_loss, '-o', label='valid')
                plt.xlabel('Epoch')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(args.save_dir, 'losses_%s.png' % args.name))
                plt.close()
                
            elif epoch - best_epoch > args.patience:
                print(f'no improvement for {args.patience+1} epochs, early stopping after {epoch+1} epochs | best epoch: {best_epoch}')
                f.write(f'no improvement after {epoch+1} epochs, early stopping...')
                break
            
            end_epoch = time.time()
            print(f'epoch {epoch} completed in {end_epoch-start_epoch:.2f} seconds')
            print('-' * 50)
            
        end_training = time.time()
        print(f'training completed in {end_training-start_training:.2f} seconds')
        print('-' * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--save-dir', type=str, default='./outputs')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--print-every', type=int, default=10000, help='print loss every n iterations')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=1, help='patience for early stopping')
    parser.add_argument('--random', type=bool, default=False,
                        help='Use randomly initialized models instead of pretrained feature extractors')
    parser.add_argument('--layer', type=str, default='layer4', help='target layer')
    parser.add_argument('--model', type=str, default='resnet50', help='target network')
    parser.add_argument('--pretrain-weights', type=str, default='IMAGENET1K_V1', help='pretrained weights')
    parser.add_argument('--refer', type=str, default='vg', choices=('vg', 'coco'), help='reference dataset')
    parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
    parser.add_argument('--name', type=str, default='', help='experiment name')
    parser.add_argument('--anno-rate', type=float, default=1., help='fraction of concepts used for supervision')
    parser.add_argument('--margin', type=float, default=1., help='hyperparameter for margin ranking loss')
    parser.add_argument('--classifier_name', type=str, default='fc', help='name of classifier layer')
    
    args = parser.parse_args()
    
    print(f'Training explainer on {args.model} {args.layer} with refer dataset {args.refer} and anno rate {args.anno_rate} for {args.epochs} epochs...\n')
    
    print(f'args: {args}\n')

    main(args)
