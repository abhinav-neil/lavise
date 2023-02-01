import numpy as np
import time
import argparse
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
import torchvision
from torchtext.vocab import GloVe
from torchvision.utils import save_image
from torch.nn.utils.rnn import pad_sequence
from image_datasets import *
from model_loader import setup_explainer

warnings.filterwarnings("ignore")   # ignore stupid dataloader warnings

def inference(args):
    method = args.method
    num_top_samples = args.imgs_per_filter
    act_dim = 7 # 7x7 activation map
    mask_transform = mask_process(act_dim) # masks transform for coco
    unseen_labels = set([]) 

    # prepare the pretrained word embedding vectors
    embedding_glove = GloVe(name='6B', dim=args.word_embedding_dim)
    embeddings = embedding_glove.vectors.T.cuda()

    # prepare the reference dataset
    if args.refer == 'vg':
        dataset = VisualGenome(root_dir='./data', transform=data_transforms['val'])
        # take the concepts learned from reference dataset
        with open("data/vg/vg_labels.pkl", 'rb') as f:
            labels = pickle.load(f)
            if args.unsupervised_concepts > 0.0:
                split_ = int(len(labels)*args.unsupervised_concepts)
                training_set = set({label:labels[label] for label in list(labels.keys())[split_:]}.keys())
            else:
                training_set = set({label:labels[label] for label in list(labels.keys())}.keys())
            print('Concepts learned from training set:')
            print(training_set)
            novel_concepts = set([])
    elif args.refer == 'coco':
        dataset = MyCocoDetection(root='./data/coco/val2017',
                                  annFile='./data/coco/annotations/instances_val2017.json',
                                  transform=data_transforms['val'])
    else:
        raise NotImplementedError
    
    print(f'number of samples in the dataset: {len(dataset)}')
    # get small subset of the dataset for testing
    indices = np.random.randint(0, len(dataset), size=1000)
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    
    def collate_fn(batch):
        if args.refer=='coco':
            imgs, annotations = zip(*batch)
            imgs = torch.stack(list(imgs), dim=0)
            return imgs, annotations
        elif args.refer=='vg':
            imgs, targets, masks, objs = zip(*batch)
            # Pad the targets and masks with zeros
            targets = pad_sequence(targets, batch_first=True)
            masks = pad_sequence(masks, batch_first=True)
            # objs = pad_sequence(objs, batch_first=True)
            # Stack the images together
            imgs = torch.stack(imgs)
        return imgs, targets, masks, objs

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # load the target model with a trained explainer
    model = setup_explainer(args, random_feature=args.random)
    if len(args.model_path) < 1:
        args.model_path = f'{args.save_dir}/{args.name}/best_model.pth.tar'
    if len(args.max_path) < 1:
        args.max_path = f'{args.save_dir}/{args.name}/max_activations.pt'
    ckpt = torch.load(args.model_path)
    print(f'loaded {args.model} trained for {ckpt["epoch"]} epochs')
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # get the max activation of each examples on the target filter
    if not os.path.exists(args.max_path):
        print('extracting max activations...')
        for k, batch in tqdm(enumerate(dataloader)):
            img = batch[0].cuda()
            x = img.clone().detach()
            for name, module in model._modules.items():
                x = module(x)
                if name==args.layer:
                    break
            activations = x.cpu().detach().numpy()
            if k == 0:
                max_activations = np.zeros((activations.shape[1],len(dataset)))
            max_activations[:,k] = np.max(activations.squeeze(0), axis=(-1, -2))
        torch.save(max_activations, args.max_path)
        print(f'activations of all filters saved to {args.max_path}')
            
    max_activations = torch.load(args.max_path)
    num_filters = max_activations.shape[0]
    print(f'{args.layer} of {args.model} has {num_filters} filters')
    if args.filters:
        filters = args.filters
    elif args.num_top_filters:
        # get top-m most activated filters
        avg_activations = np.mean(max_activations, axis=1)
        top_m_filters = np.argsort(avg_activations)[-args.num_top_filters:][::-1]
        print(f'top {args.num_top_filters} most activated filters for dataset {args.refer}: {top_m_filters}')
        filters = top_m_filters # we only do inference on the top m filters
    else:
        filters = np.arange(num_filters)
    # sort images by their max activations
    sorted_samples = np.argsort(-max_activations, axis=1)
    del max_activations # free memory
    
    # get activation thresholds
    def get_act_thresholds(filters):
        all_activations = []
        for k, batch in enumerate(dataloader):
            img = batch[0].cuda()
            x = img.clone().detach()
            for name, module in model._modules.items():
                x = module(x)
                if name==args.layer:
                    break    # stop forward pass after target layer
            act = x.cpu().detach().numpy()
            act = act[:, filters, :, :].transpose(1, 0, 2, 3).reshape(len(filters), -1)
            all_activations.append(act)
        end_time_loop = time.time()
        print(f'completed in {end_time_loop - start_time_loop:.2f}s')
        all_activations = np.concatenate(all_activations, axis=1)
        act_thresholds = np.percentile(all_activations, args.act_q, axis=1)
        return act_thresholds
    
    act_thresh_path = f'outputs/{args.name}/act_thresholds.json'
    if not os.path.exists(act_thresh_path):
        print(f'computing activation thresholds...\n')
        max_filters = 256 # split the filters into chunks to avoid OOM
        filters_list = np.array_split(filters, len(filters)//max_filters)
        start_time_loop = time.time()
        act_thresholds = []
        for idx, filters_subset in enumerate(filters_list):
            print(f'subset {idx+1}/{len(filters_list)}\n')
            act_thresholds_subset = get_act_thresholds(filters_subset)
            act_thresholds += act_thresholds_subset.tolist()
        end_time_loop = time.time()
        print(f'computed activation thresholds for {len(filters)} filters in {end_time_loop-start_time_loop:.2f}s')
        act_thresholds = {int(f): act_thresholds[i] for i, f in enumerate(filters)}
        with open(act_thresh_path, 'w') as file:
            json.dump(act_thresholds, file)
        print(f'activation thresholds saved to {act_thresh_path}')
        
    print(f'loading activation thresholds from {act_thresh_path}...')
    # act_threshold = np.load(act_thresh_path, allow_pickle=True)[filters]
    with open(act_thresh_path, 'r') as file:
        act_thresholds = json.load(file)
    # convert keys to int
    act_thresholds = {int(k): v for k, v in act_thresholds.items()}
    
    # inference 
    results = {'method': args.method, 'num_filters': len(filters), 
               'imgs_per_filter': args.imgs_per_filter, 'words_per_img': args.words_per_img,
               **{int(f): {k: {} for k in ['ground_truths', 'predictions', 'recall@5', 'recall@10', 'recall@20']} for f in filters},
               'avg_recalls': dict.fromkeys([5, 10, 20], 0.)}
    
    print(f'running inference on {len(filters)} filters...\n')
    start_infer_loop = time.time()
    for filter_idx, _filter in enumerate(filters):
        print('-'*50)
        with torch.no_grad():
            start = time.time()
            print(f'explaining filter {_filter} with {num_top_samples} top activated images... [{filter_idx+1}/{len(filters)}]\n')
            filter_dataset = torch.utils.data.Subset(dataset, sorted_samples[_filter, :num_top_samples])
            filter_dataloader = torch.utils.data.DataLoader(filter_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
            weights = 0
            for i, batch in enumerate(filter_dataloader):
                if args.refer == 'vg':
                    data_, targets, masks, objs = batch[0].cuda(), batch[1].squeeze(0), batch[2].squeeze(0), batch[3]
                elif args.refer =='coco':
                     data_, annotations = batch[0].cuda(), batch[1] 
                     masks = [mask_transform(d['mask'].squeeze(0)) for d in annotations]
                     objs = [tuple(d['object']) for d in annotations]
                x = data_.clone()
                for name, module in model._modules.items():
                    x = module(x)
                    if name==args.layer:
                        activation = x.detach().cpu().numpy()
                        break
                act_f = activation[:, _filter, :, :].squeeze()
                act_region = np.where(act_f > act_thresholds[_filter], 1, 0)
                # compute intersection-over-union (IoU) between the mask and the activation region
                ground_truths = set()
                iou_scores = []
                for mask, obj in zip(masks, objs):
                    mask = mask.cpu().numpy()
                    intersection = np.sum(np.logical_and(act_region, mask))
                    union = np.sum(np.logical_or(act_region, mask))
                    iou = intersection / union
                    if iou > args.mask_threshold:
                        ground_truths.add(obj[0])
                    iou_scores.append(iou)
                if len(ground_truths) == 0:
                    ground_truths.add(None) # prevent division by zero 
                # print(f'\nground truth concepts for filter {filter} and image {i}: {ground_truths}')
                results[_filter]['ground_truths'][i] = list(ground_truths)
                iou_scores = np.array(iou_scores)
                act_f_upsampled = cv2.resize(act_f, (224, 224))
                weight = np.amax(act_f_upsampled)
                if weight <= 0.:
                    continue

                # interpret the explainer's output with the specified method
                predict = explain(method, model, data_, activation, act_f, act_f_upsampled, act_thresholds[_filter])
                predict_score = torch.mm(predict, embeddings) / \
                                torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                        torch.sqrt(torch.sum(embeddings ** 2, dim=0, keepdim=True)))
                sorted_predict_score, sorted_predict = torch.sort(predict_score, dim=1, descending=True)
                sorted_predict = sorted_predict[0, :].detach().cpu().numpy()
                select_rank = np.repeat(sorted_predict[:args.words_per_img], int(weight))

                if weights == 0:
                    filter_rank = select_rank
                else:
                    filter_rank = np.concatenate((filter_rank, select_rank))
                weights += weight
                
                # compute recall (ioU) between the predicted concepts and the ground truth concepts
                # with open('data/entities.txt') as file:
                    # all_labels = [line.rstrip() for line in file]
                (values, counts) = np.unique(filter_rank, return_counts=True)
                ind = np.argsort(-counts)
                sorted_predict_words = []
                for ii in ind[:args.words_per_img]:
                    word = embedding_glove.itos[int(values[ii])].replace("/",".") # in case there's a / in the word 
                    # if word in all_labels:
                    if word not in training_set:
                        novel_concepts.add(word)
                        word = '(' + word + ')'
                    sorted_predict_words.append(word)
                results[_filter]['predictions'][i] = sorted_predict_words
                print(f'predicted concepts for filter {_filter} and image {i}: {sorted_predict_words}')

                #if args.unsupervised_concepts > 0.0: 
                #    novel_concepts = set(sorted_predict_words).intersection(unseen_labels)
                #    print(f'novel concepts found on image {i}: {novel_concepts}')

                # results[_filter]['recall'][i] = len(ground_truths.intersection(set(sorted_predict_words))) / len(ground_truths)
                # print(f'recall for filter {_filter} and image {i}: {results[_filter]["recall"][i]}')
                for k in [5, 10, 20]:
                    # if len(ground_truths) >= k:
                    results[_filter][f'recall@{k}'][i] = len(ground_truths.intersection(set(sorted_predict_words[:k]))) / len(ground_truths)
                    print(f"recall@{k} for filter {_filter} and image {i}: {results[_filter][f'recall@{k}'][i]}")
                # visualize
                if i < args.viz_per_filter:
                    # heatmaps
                    # print(f'visualizing heatmaps for filter {_filter} and image {i}...\n')
                    words_per_caption = min(args.words_per_img, 5) # limit caption to 5 words
                    heatmaps_dir = f'outputs/{args.name}/{args.method}/heatmaps'
                    if not os.path.exists(heatmaps_dir):
                        os.makedirs(heatmaps_dir)
                    # max_weights = np.zeros(3)
                    # top_k_heatmaps = [0, 0, 0]
                    # for idx, weight_max in enumerate(max_weights):
                        # if weight > weight_max:
                            # max_weights[idx] = weights
                    viz_img = data_.cpu().permute(0,2,3,1)
                    viz_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                    viz_img = np.array(viz_img.permute(0,3,1,2).squeeze(0))
                    activation = act_f_upsampled/act_f_upsampled.max()
                    #activation = act_f_upsampled
                    activation = np.repeat(np.expand_dims(activation, 0), 3, axis=0)
                    heatmap_vis = torch.tensor(0.8 * activation + 0.2 * viz_img)
                    # top_k_heatmaps[idx] = torch.tensor(heatmap_vis)
                    torchvision.utils.save_image(heatmap_vis, f'{heatmaps_dir}/f={_filter}_img={i}_{"_".join(sorted_predict_words[:words_per_caption])}.png' )
                    # filter_images_dir = f'outputs/{args.name}/{args.method}/images'
                    # if not os.path.exists(filter_images_dir):
                    #     os.makedirs(filter_images_dir)
                    # filter_image_array = torchvision.utils.make_grid(top_k_heatmaps)
                    # caption = f'_{args.method}_{_filter}' + f'_'.join(sorted_predict_words) 
                    # torchvision.utils.save_image(filter_image_array, f'{filter_images_dir}/{caption}.png')
            # results[_filter]['ground_truths'] = list(ground_truths)
            end = time.time()
            print(f'\nelapsed time: {end - start}\n')
            # results[filter]['recall']['avg'] = np.nanmean(list(results[filter]['recall'].values()))
            # print(f'avg recall for filter {filter} is {results[filter]["recall"]["avg"]:.3f}')
            for k in [5, 10, 20]:
                results[_filter][f'recall@{k}']['avg'] = np.nanmean(list(results[_filter][f'recall@{k}'].values()))
                print(f'avg recall@{k} for filter {_filter} is {results[_filter][f"recall@{k}"]["avg"]:.3f}')
        
    print('-' * 50)
    end_infer_loop = time.time()
    print(f'completed inference for {len(filters)} filters in {end_infer_loop - start_infer_loop} seconds')
    print(f'novel concepts found: {len(novel_concepts)}')
    print(novel_concepts)
    # results['avg_recalls']['all'] = np.nanmean([results[filter]['recall']['avg'] for filter in filters])
    # print(f'avg recall (IoU) for {len(filters)} filters is {results["avg_recalls"]["all"]:.3f}')
    for k in [5, 10, 20]:
        results[f'avg_recalls'][k] = np.nanmean([results[_filter][f'recall@{k}']['avg'] for _filter in filters])
        print(f'avg recall@{k} (IoU) for {len(filters)} filters is {results["avg_recalls"][k]:.3f}')
    results_path = f'outputs/{args.name}/{args.method}/results.json'
    # if not os.path.exists(results_path):
    with open(results_path, 'w') as f:
        json.dump(results, f)
        print(f'\nresults for {len(filters)} filters saved to {results_path}')
    print('-' * 50)
    
def explain(method, model, data_, activation, act_f, act_f_upsampled, act_threshold):
    img = data_.detach().cpu().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))
    if method == 'original':
        # original image
        data = data_.clone().requires_grad_(True)
        predict = model(data)
    elif method == 'projection':
        # filter attention projection
        filter_embed = torch.tensor(
            np.mean(activation * act_f / (np.sum(act_f ** 2, axis=(0, 1)) ** .5), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    elif method == 'image':
        # image masking
        data = img * (act_f_upsampled[:, :, None] > act_threshold)
        data = torch.tensor(np.transpose(data, (2, 0, 1))).unsqueeze(0).cuda()
        predict = model(data)
    elif method == 'activation':
        # activation masking
        filter_embed = torch.tensor(np.mean(activation * (act_f > act_threshold), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    else:
        raise NotImplementedError

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=str, default='layer4', help='target layer')
    parser.add_argument('--filters', type=int, default=None, nargs='+', help='list of index of the target filters')
    parser.add_argument('--num_top_filters', type=int, default=None, help='number of top most activated filters per dataset. Defaults to all filters')
    parser.add_argument('--method', type=str, default='projection',
                        choices=('original', 'image', 'activation', 'projection'),
                        help='method used to explain the target filter')
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--refer', type=str, default='vg', choices=('vg', 'coco'), help='reference dataset')
    # parser.add_argument('--num-output', type=int, default=10,
                        # help='number of words used to explain the target filter')
    parser.add_argument('--save-dir', type=str, default='./outputs')
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
                        help='')
    parser.add_argument('--act_q', type=float, default=99.5, help='quantile for activation threshold')
    # if filter activation projection is used
    parser.add_argument('--words_per_img', type=int, default=20,
                        help='number of words used to explain the target filter')
    parser.add_argument('--imgs_per_filter', type=int, default=25,
                        help='number of top activated images used to explain each filter')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    # parser.add_argument('--visualize', type=bool, default=True, help='visualize the predictions')
    parser.add_argument('--viz_per_filter', type=int, default=2, help='number of images to visualize per filter')
    parser.add_argument('--unsupervised_concepts', type=float, default=0.0, 
                        help='percentage of dataset to during feature extraction')
    

    args = parser.parse_args()
    print(f'\nrunning inference on {args.refer} {args.model} {args.layer} with method {args.method}\n')
    print(f'{args}\n')

    inference(args)

