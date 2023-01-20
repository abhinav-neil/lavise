import cv2
import time
import argparse
from image_datasets import *
from model_loader import setup_explainer
from torchtext.vocab import GloVe
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def inference(args):
    f = args.f
    method = args.method
    num_top_samples = args.p

    # prepare the pretrained word embedding vectors
    embedding_glove = GloVe(name='6B', dim=args.word_embedding_dim)
    embeddings = embedding_glove.vectors.T.cuda()

    # prepare the reference dataset
    if args.refer == 'vg':
        dataset = VisualGenome(transform=data_transforms['val'])
    elif args.refer == 'coco':
        dataset = MyCocoDetection(root='./data/coco/val2017',
                                  annFile='./data/coco/annotations/instances_val2017.json',
                                  transform=data_transforms['val'])
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # load the target model with a trained explainer
    model = setup_explainer(args, random_feature=args.random)
    if len(args.model_path) < 1:
        args.model_path = 'outputs/' + args.name + '/ckpt_tmp.pth.tar'
    if len(args.max_path) < 1:
        args.max_path = 'outputs/' + args.name + '/act_max_{}.pt'.format(args.method)
    ckpt = torch.load(args.model_path)
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # get the max activation of each examples on the target filter
    #max_activations = np.zeros(len(dataset))
    if not os.path.exists(args.max_path):
        print('extracting max activations...')
        for k, batch in enumerate(dataloader):
            img = batch[0].cuda()
            x = img.clone().detach()
            for name, module in model._modules.items():
                x = module(x)
                if name is args.layer:
                    break
            x = x.cpu().detach().numpy()
            if k == 0:
                max_activations = np.zeros((x.shape[1],len(dataset)))
            max_activations[:,k] = np.max(x.squeeze(0), axis=(-1, -2))
            # max_activations[k] = np.max(x, axis=(-1, -2))[f]
        torch.save(max_activations, args.max_path)
        print('activations of filters saved!')
    max_activations = torch.load(args.max_path)

    # sort images by their max activations
    sorted_samples = np.argsort(-max_activations, axis=1)

    # load the activation threshold
    #threshold = np.load(args.thresh_path)[f]
    threshold = args.mask_threshold

    with torch.no_grad():
        start = time.time()
        print('explaining filter %d with %d top activated images' % (f, num_top_samples))
        filter_dataset = torch.utils.data.Subset(dataset, sorted_samples[f, :num_top_samples])
        filter_dataloader = torch.utils.data.DataLoader(filter_dataset, batch_size=1,
                                                        shuffle=False, num_workers=0)
        weights = 0
        for i, batch in enumerate(filter_dataloader):
            if not batch[1]:
                continue
            data_, annotation = batch[0].cuda(), batch[1]
            x = data_.clone()
            for name, module in model._modules.items():
                x = module(x)
                if name is args.layer:
                    activation = x.detach().cpu().numpy()
                    break
            c = activation[:, f, :, :]
            c = c.reshape(7, 7)
            xf = cv2.resize(c, (224, 224))
            weight = np.amax(c)
            if weight <= 0.:
                continue

            # interpret the explainer's output with the specified method

            # these goobers really gave the input wrong to their own function
            # fun fact, it took way too long for me to figure this out
            # predict = explain(model, data_, method, activation, c, xf, threshold)
            predict = explain(method, model, data_, activation, c, xf, threshold)
            predict_score = torch.mm(predict, embeddings) / \
                            torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                     torch.sqrt(torch.sum(embeddings ** 2, dim=0, keepdim=True)))
            sorted_predict_score, sorted_predict = torch.sort(predict_score, dim=1, descending=True)
            sorted_predict = sorted_predict[0, :].detach().cpu().numpy()
            select_rank = np.repeat(sorted_predict[:args.s], int(weight))

            if weights == 0:
                filter_rank = select_rank
            else:
                filter_rank = np.concatenate((filter_rank, select_rank))

            weights += weight

            # VISUALIZE
            max_weights = np.zeros(3)
            top_k_heatmaps = [0, 0, 0]
            for idx, weight_max in enumerate(max_weights):
                if weight > weight_max:
                    max_weights[idx] = weights
                    viz_img = data_.cpu().permute(0,2,3,1)
                    viz_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                    viz_img = np.array(viz_img.permute(0,3,1,2).squeeze(0))
                    activation = xf/xf.max()
                    activation = np.repeat(np.expand_dims(activation, 0), 3, axis=0)
                    heatmap_vis = torch.tensor(0.6 * activation + (1-0.6) * viz_img)
                    # print(type(heatmap_vis))
                    # print(type(heatmap_vis.permute(1,2,0)))
                    top_k_heatmaps[idx] = heatmap_vis
                    #torchvision.utils.save_image(heatmap_vis, 'outputs/' + args.name + '/heatmaps.png')
                    #filter_image_array = torchvision.utils.make_grid(top_k_heatmaps)
                    #torchvision.utils.save_image(filter_image_array, 'outputs/' + args.name + '/filter_grid.png')
                    #plt.imshow(heatmap_vis.transpose(1,2,0))
                    #plt.show()

        with open('data/entities.txt') as file:
            all_labels = [line.rstrip() for line in file]
        (values, counts) = np.unique(filter_rank, return_counts=True)
        ind = np.argsort(-counts)
        sorted_predict_words = []
        for ii in ind[:args.num_output]:
            word = embedding_glove.itos[int(values[ii])]
            if word in all_labels:
                sorted_predict_words.append(word)

        end = time.time()
        print('Elasped Time: %f s' % (end - start))
    filter_image_array = torchvision.utils.make_grid(top_k_heatmaps)
    caption = "{}|f:{}|{}_{}_{}".format(args.method, f, sorted_predict_words[0], sorted_predict_words[1], sorted_predict_words[2])
    torchvision.utils.save_image(filter_image_array, 'outputs/' + args.name + '/{}.png'.format(caption))
    return sorted_predict_words


def explain(method, model, data_, activation, c, xf, threshold):
    img = data_.detach().cpu().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))
    if method == 'original':
        # original image
        data = data_.clone().requires_grad_(True)
        predict = model(data)
    elif method == 'projection':
        # filter attention projection
        filter_embed = torch.tensor(
            np.mean(activation * c / (np.sum(c ** 2, axis=(0, 1)) ** .5), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    elif method == 'image':
        # image masking
        data = img * (xf[:, :, None] > threshold)
        data = torch.tensor(np.transpose(data, (2, 0, 1))).unsqueeze(0).cuda()
        predict = model(data)
    elif method == 'activation':
        # activation masking
        filter_embed = torch.tensor(np.mean(activation * (c > threshold), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    else:
        raise NotImplementedError

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=str, default='layer4', help='target layer')
    #parser.add_argument('--f', type=int, help='index of the target filter')
    # just chose a random default
    parser.add_argument('--f', type=int, default=50, help='list of index of the target filters')
    parser.add_argument('--method', type=str, default='projection',
                        choices=('original', 'image', 'activation', 'projection'),
                        help='method used to explain the target filter')
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--refer', type=str, default='vg', choices=('vg', 'coco'), help='reference dataset')
    parser.add_argument('--num-output', type=int, default=10,
                        help='number of words used to explain the target filter')
    parser.add_argument('--model-path', type=str, default='', help='path to load the target model')
    parser.add_argument('--thresh-path', type=str, help='path to save/load the thresholds')
    parser.add_argument('--max-path', type=str, default='',
                        help='path to save/load the max activations of all examples')

    # parser arguments because these need to be present for setup explainer
    parser.add_argument('--random', type=bool, default=False,
                        help='Use randomly initialized models instead of pretrained feature extractors')
    parser.add_argument('--model', type=str, default='resnet50', help='target network')
    parser.add_argument('--classifier_name', type=str, default='fc', help='name of classifier layer')
    parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
    parser.add_argument('--name', type=str, default='random_test', help='experiment name')
    parser.add_argument('--mask_threshold', type=float, default=0.04,
                        help='path to save/load the thresholds')

    # if filter activation projection is used
    parser.add_argument('--s', type=int, default=5,
                        help='number of semantics contributed by each top activated image')
    parser.add_argument('--p', type=int, default=25,
                        help='number of top activated images used to explain each filter')

    args = parser.parse_args()
    print(args)

    inference(args)
