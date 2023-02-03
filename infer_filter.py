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
from torch.utils.data.dataset import random_split
from image_datasets import *
from model_loader import setup_explainer

warnings.filterwarnings("ignore")   # ignore dataloader warnings

def inference(args):
    method = args.method
    num_top_samples = args.imgs_per_filter
    act_dims = {'layer4': 7, 'layer3': 14, 'layer2': 28, 'layer1': 56}
    act_dim = act_dims[args.layer] # 7x7 activation map
    mask_transform = mask_process(act_dim) # masks transform for coco

    # prepare the pretrained word embedding vectors
    embedding_glove = GloVe(name='6B', dim=args.word_embedding_dim)
    embeddings = embedding_glove.vectors.T.cuda()

    # prepare the reference dataset
    if args.refer == 'vg':
        dataset = VisualGenome(root_dir='./data', transform=data_transforms['val'])
        torch.manual_seed(0)
        with open("./data/vg/vg_labels.pkl", 'rb') as f:
            labels = pickle.load(f)
        label_index = []
        for label in labels:
            label_index.append(embedding_glove.stoi[label])
        labels = np.array(list(labels.keys()))
        np.random.seed(0)
        train_label_index = np.random.choice(range(len(label_index)), int(len(label_index) * args.anno_rate), replace=False)
        annotated_concepts = set(labels)
        seen_concepts = set(labels[train_label_index])
        unseen_concepts = annotated_concepts - seen_concepts
        print(f'the dataset has {len(annotated_concepts)} annotated concepts of which {len(seen_concepts)} are seen during training and {len(unseen_concepts)} are unseen\n')
        learned_concepts = set()
        novel_learned_concepts = set()

    elif args.refer == 'coco':
        dataset = MyCocoDetection(root='./data/coco/val2017',
                                  annFile='./data/coco/annotations/instances_val2017.json',
                                  transform=data_transforms['val'])
        annotated_concepts = set([])
        unseen_concepts = set([])
        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        label_embedding = torch.load(label_embedding_file)
        annotated_concepts = list(label_embedding['itos'].keys())
        seen_concepts = annotated_concepts
        print(f'the dataset has {len(annotated_concepts)} annotated concepts of which {len(seen_concepts)} are seen during training and {len(unseen_concepts)} are unseen\n')
        learned_concepts = set()
        novel_learned_concepts = set()
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
        start = time.time()
        for k, batch in enumerate(dataloader):
            img = batch[0].cuda()
            x = img.clone().detach()
            for name, module in model._modules.items():
                x = module(x)
                if name==args.layer:
                    break
            activations = x.cpu().detach().numpy()
            if k == 0:
                max_activations = np.zeros((activations.shape[1],len(dataset)))
            max_activations[:,k*args.batch_size:(k+1)*args.batch_size] = np.max(activations, axis=(-1, -2)).T
        torch.save(max_activations, args.max_path)
        print(f'activations of all filters saved to {args.max_path}')
        end = time.time()
        print(f'completed in {end-start:.2f} seconds\n')
            
    max_activations = torch.load(args.max_path)
    num_filters = max_activations.shape[0]
    print(f'{args.layer} of {args.model} has {num_filters} filters\n')
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
        max_filters = 128 # split the filters into chunks to avoid OOM
        if len(filters) > max_filters:
            filters_list = np.array_split(filters, len(filters)//max_filters)
        else:
            filters_list = [filters]
        start_time_loop = time.time()
        act_thresholds = []
        for idx, filters_subset in enumerate(filters_list):
            print(f'subset {idx+1}/{len(filters_list)}')
            act_thresholds_subset = get_act_thresholds(filters_subset)
            act_thresholds += act_thresholds_subset.tolist()
        end_time_loop = time.time()
        print(f'computed activation thresholds for {len(filters)} filters in {end_time_loop-start_time_loop:.2f}s')
        act_thresholds = {int(f): act_thresholds[i] for i, f in enumerate(filters)}
        with open(act_thresh_path, 'w') as file:
            json.dump(act_thresholds, file)
        print(f'activation thresholds saved to {act_thresh_path}')
        
    print(f'loading activation thresholds from {act_thresh_path}...')
    with open(act_thresh_path, 'r') as file:
        act_thresholds = json.load(file)
    # convert keys to int
    act_thresholds = {int(k): v for k, v in act_thresholds.items()}
    
    # inference 
    output_path = f'outputs/{args.name}/{args.method}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results = {'method': args.method, 'num_filters': len(filters), 
               'imgs_per_filter': args.imgs_per_filter, 'words_per_img': args.words_per_img,
               **{int(f): {k: {} for k in ['ground_truths', 'predictions', 'recall@5', 'recall@10', 'recall@20', 'recall@5 (unseen)', 'recall@10 (unseen)', 'recall@20 (unseen)']} for f in filters},
               'avg_recalls': dict.fromkeys([5, 10, 20, '5 (unseen)', '10 (unseen)', '20 (unseen)'], 0.)}
    
    print(f'running inference on {len(filters)} filters...\n')
    start_infer_loop = time.time()
    for filter_idx, filter in enumerate(filters):
        print('-'*50)
        with torch.no_grad():
            start = time.time()
            print(f'explaining filter {filter} with {num_top_samples} top activated images... [{filter_idx+1}/{len(filters)}]\n')
            filter_dataset = torch.utils.data.Subset(dataset, sorted_samples[filter, :num_top_samples])
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
                act_f = activation[:, filter, :, :].squeeze()
                act_region = np.where(act_f > act_thresholds[filter], 1, 0)
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
                    print(f'\nno ground truth concepts for filter {filter} and image {i}')
                    continue
                ground_truths_unseen = ground_truths - seen_concepts
                print(f'\nground truth concepts for filter {filter} and image {i}: {list(ground_truths)}')
                results[filter]['ground_truths'][i] = list(ground_truths)
                iou_scores = np.array(iou_scores)
                act_f_upsampled = cv2.resize(act_f, (224, 224))
                weight = np.amax(act_f_upsampled)
                if weight <= 0.:
                    continue

                # interpret the explainer's output with the specified method
                predict = explain(method, model, data_, activation, act_f, act_f_upsampled, act_thresholds[filter])
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
                
                (values, counts) = np.unique(filter_rank, return_counts=True)
                ind = np.argsort(-counts)
                sorted_predict_words = []
                for ii in ind[:args.words_per_img]:
                    word = embedding_glove.itos[int(values[ii])]
                    if word in annotated_concepts and word not in seen_concepts:
                        novel_learned_concepts.add(word)
                    sorted_predict_words.append(word)
                learned_concepts.update(sorted_predict_words)
                results[filter]['predictions'][i] = sorted_predict_words
                print(f'predicted concepts for filter {filter} and image {i}: {sorted_predict_words}')

                for k in [5, 10, 20]:
                    results[filter][f'recall@{k}'][i] = len(ground_truths.intersection(set(sorted_predict_words[:k]))) / len(ground_truths)
                    print(f"recall@{k} for filter {filter} and image {i}: {results[filter][f'recall@{k}'][i]}")
                    results[filter][f'recall@{k} (unseen)'][i] = len(ground_truths_unseen.intersection(set(sorted_predict_words[:k]))) / len(ground_truths_unseen) if len(ground_truths_unseen) > 0 else np.nan
                    print(f"recall@{k} (unseen) for filter {filter} and image {i}: {results[filter][f'recall@{k} (unseen)'][i]}")
                    
                # visualize
                if i < args.viz_per_filter:
                    # heatmaps
                    print(f'visualizing heatmaps for filter {filter} and image {i}...\n')
                    words_per_caption = min(args.words_per_img, 5) # limit caption to 5 words
                    heatmaps_dir = f'{output_path}/heatmaps'
                    if not os.path.exists(heatmaps_dir):
                        os.makedirs(heatmaps_dir)
                    viz_img = data_.cpu().permute(0,2,3,1)
                    viz_img = np.array(viz_img.permute(0,3,1,2).squeeze(0))
                    activation = act_f_upsampled / act_f_upsampled.max()
                    activation = np.repeat(np.expand_dims(activation, 0), 3, axis=0)
                    heatmap_vis = torch.tensor(args.heatmap_opacity * activation + (1 - args.heatmap_opacity) * viz_img)
                    caption = f"f={filter}_img={i}_{'_'.join(['('+word+')' if word in novel_learned_concepts else word for word in sorted_predict_words[:words_per_caption]])}"
                    torchvision.utils.save_image(heatmap_vis, f'{heatmaps_dir}/{caption}.png' )

            end = time.time()
            print(f'\nelapsed time: {(end - start):.2f} seconds')
            for k in [5, 10, 20]:
                results[filter][f'recall@{k}']['avg'] = np.nanmean(list(results[filter][f'recall@{k}'].values()))
                print(f'avg recall@{k} for filter {filter} is {results[filter][f"recall@{k}"]["avg"]:.3f}')
                results[filter][f'recall@{k} (unseen)']['avg'] = np.nanmean(list(results[filter][f'recall@{k} (unseen)'].values()))
                print(f'avg recall@{k} (unseen) for filter {filter} is {results[filter][f"recall@{k} (unseen)"]["avg"]:.3f}')
        
    print('-' * 50)
    end_infer_loop = time.time()
    print(f'completed inference for {len(filters)} filters in {(end_infer_loop - start_infer_loop):.2f} seconds')    

    for k in [5, 10, 20]:
        results[f'avg_recalls'][k] = np.nanmean([results[filter][f'recall@{k}']['avg'] for filter in filters])
        print(f'avg recall@{k} (IoU) for {len(filters)} filters is {results["avg_recalls"][k]:.3f}')
        results[f'avg_recalls'][f'{k} (unseen)'] = np.nanmean([results[filter][f'recall@{k} (unseen)']['avg'] for filter in filters])
        print(f'avg recall@{k} (unseen) (IoU) for {len(filters)} filters is {results["avg_recalls"][f"{k} (unseen)"]:.3f}')
    
    # novel concepts
    results['concepts'] = {'annotated': list(annotated_concepts), 'seen': list(seen_concepts), 'unseen': list(unseen_concepts),
                           'learned': list(learned_concepts), 'novel learned': list(novel_learned_concepts),
                           '# annotated': len(annotated_concepts), '# seen': len(seen_concepts), '# unseen': len(unseen_concepts), 
                           '# learned': len(learned_concepts), '# novel learned': len(novel_learned_concepts),
                           'fraction discovered': len(novel_learned_concepts) / len(unseen_concepts) if len(unseen_concepts) > 0 else 0}
    print(f'novel concepts discovered: {novel_learned_concepts}\n')
    print(f'learned {len(learned_concepts)} concepts and discovered {len(novel_learned_concepts)} novel concepts out of {len(unseen_concepts)} unseen concepts ({results["concepts"]["fraction discovered"]:.3f})\n')
    
    results_path = f'{output_path}/results.json'
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
    parser.add_argument('--save-dir', type=str, default='./outputs')
    parser.add_argument('--model-path', type=str, default='', help='path to load the target model')
    parser.add_argument('--thresh-path', type=str, help='path to save/load the thresholds')
    parser.add_argument('--max-path', type=str, default='',
                        help='path to save/load the max activations of all examples')
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
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers for dataloader')
    parser.add_argument('--viz_per_filter', type=int, default=1, help='number of images to visualize per filter')
    parser.add_argument('--heatmap_opacity', type=float, default=0.75, help='opacity of the heatmap')
    parser.add_argument('--anno_rate', type=float, default=1.0, help='percentage of reference dataset that was seen during training')
    

    args = parser.parse_args()
    print(f'\nRunning inference on {args.refer} {args.model} {args.layer} with method {args.method}\n')
    print(f'{args}\n')

    inference(args)

