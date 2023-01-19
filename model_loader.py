import torch
import torchvision
import torch.nn as nn


def setup_explainer(settings, hook_fn=None, random_feature=False):
    if random_feature:
        model = torchvision.models.__dict__[settings.model](pretrained=False)
    # elif settings.pretrain is None:
    else:
        model = torchvision.models.__dict__[settings.model](pretrained=True)
    # else:
    #     checkpoint = torch.load(settings.pretrain)
    #     if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
    #         state_dict = checkpoint['state_dict']
    #         model = torchvision.models.__dict__[settings.model](num_classes=settings.num_classes)
    #         model.load_state_dict(state_dict)
    #     else:
    #         model = checkpoint

    for param in model.parameters():
        param.requires_grad = False

    target_index = list(model._modules).index(settings.layer)
    classifier_index = list(model._modules).index(settings.classifier_name)
    feature_dim = list(model._modules[settings.layer]._modules.values())[-1].conv3.out_channels
    for module_name in list(model._modules)[target_index + 1:classifier_index]:
        if module_name[-4:] == 'pool':
            continue
        else:
            model._modules[module_name] = nn.Identity()

    if settings.model[:6] == 'resnet':
        # feature_dim = model._modules[settings.classifier_name].in_features
        model._modules[settings.classifier_name] = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.1),
            nn.Linear(
                in_features=feature_dim,
                out_features=feature_dim,
                bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.1),
            nn.Linear(in_features=feature_dim,
                      out_features=settings.word_embedding_dim,
                      bias=True))
    else:
        raise NotImplementedError

    if settings.pretrain is not None:
        checkpoint = torch.load(settings.pretrain)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)

    if hook_fn is not None:
        model._modules.get(settings.layer).register_forward_hook(hook_fn)

    model.cuda()
    return model

