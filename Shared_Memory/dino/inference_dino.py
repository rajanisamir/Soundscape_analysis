import os
import sys
import argparse
import datetime
import time
import re
from pathlib import Path

import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits



class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path





def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size, interpolation=3),
        #pth_transforms.CenterCrop(224),
        # pth_transforms.CenterCrop((1700,2048)),
        pth_transforms.Grayscale(num_output_channels=1),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.5, ), (0.5, )),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ReturnIndexDataset(args.data_path, transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    start_time = time.time()
    # ============ extract features ... ============
    print("Extracting features ...")
    features, file_names, file_paths, attentional_maps = extract_features(model, data_loader, args.use_cuda)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Inference time {}'.format(total_time_str))

    if utils.get_rank() == 0:
        features = nn.functional.normalize(features, dim=1, p=2)

    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(features.cpu(), os.path.join(args.dump_features, "feat.pth"))
        torch.save(file_names.cpu(), os.path.join(args.dump_features, "file_name.pth"))
        torch.save(file_paths.cpu(), os.path.join(args.dump_features, "file_path.pth"))
        torch.save(attentional_maps.cpu(), os.path.join(args.dump_features, "att_map.pth"))
        print(f"Features and attentional maps with their corresponding file names are saved in {args.dump_features}!")
    return features





@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    file_names = None
    file_paths = None
    attentional_maps = None
    if args.inference_up_to:
        cumulative_indexes = torch.zeros(len(data_loader.dataset), dtype=torch.bool)
        counter = 0

    printing=True
    for images, index, path in metric_logger.log_every(data_loader, 10):
        if args.inference_up_to and counter >= args.inference_up_to:
                break

        # move images to gpu
        images = images.cuda(non_blocking=True)

        #images = pth_transforms.functional.crop(images,500,0,500,1024)
        # images = pth_transforms.functional.crop(images,550,0,200,1548)
        # make the image divisible by the patch size
        w, h = images.shape[-2] - images.shape[-2] % args.patch_size, images.shape[-1] - images.shape[-1] % args.patch_size
        images = images[:, :, :w, :h]
        if printing and dist.get_rank() == 0:
            printing=False
            print('images shape is ', images.shape)

        index = index.cuda(non_blocking=True)

        # get file names
        local_file_names = []
        local_file_paths = []
        for i in range(args.batch_size_per_gpu):
            # file_name = re.sub("[^0-9]", "", os.path.basename(path[i])[25:]).zfill(15)
            # file_name = re.sub("[^0-9]", "", os.path.basename(path[i])[:25]) + file_name
            # file_name = re.sub("[^0-9]", "", os.path.basename(path[i])[19:]).zfill(5) 
            # file_name = re.sub("[^0-9]", "", os.path.basename(path[i])[:19]) + file_name
            # file_name = re.sub("[^0-9]", "", os.path.basename(path[i]))
            # file_name = np.array(list(file_name), dtype=int)
            file_name = os.path.basename(path[i]).zfill(15)
            file_name = np.array([ord(x) for x in file_name], dtype=int)
            local_file_names.append(file_name)


            p=Path(path[i])
            file_path = ''
            for j in reversed(range(args.directory_depth)):
                if j+1 == args.directory_depth:
                    file_path = file_path + p.parts[-(j+1)]
                else:
                    file_path = file_path + '/' + p.parts[-(j+1)]

            if 'justify_string' not in locals():
                if dist.get_rank() == 0:
                    justify_string = torch.tensor(len(file_path) + 20)
                    justify_string = justify_string.cuda(non_blocking=True)
                    torch.distributed.broadcast(justify_string, src=0, async_op=False)
                    justify_string = justify_string.item()
                else:
                    justify_string = torch.tensor(0)
                    justify_string = justify_string.cuda(non_blocking=True)
                    torch.distributed.broadcast(justify_string, src=0, async_op=False)
                    justify_string = justify_string.item()



            # print('justify_string: ', justify_string, ' len(file_path)', len(file_path), ' in rank ', dist.get_rank())
            if len(file_path) > justify_string:
                raise RuntimeError('Incompatible path length, please increase "justify_string"', 'justify_string: ', justify_string, ' len(file_path)', len(file_path), ' in rank ', dist.get_rank())
            else:
                file_path = file_path.ljust(justify_string, '~')

            #print(file_path)
            file_path = np.array([ord(x) for x in file_path], dtype=int)
            local_file_paths.append(file_path)
            #print(file_path)

        local_file_names = np.array(local_file_names)
        #print(local_file_names)
        local_file_names = torch.from_numpy(local_file_names).float()

        # move local file names to gpu
        local_file_names = local_file_names.cuda(non_blocking=True)
        #print('local_file_names shape is ', local_file_names.shape)




        local_file_paths = np.array(local_file_paths)
        #print(local_file_paths)
        local_file_paths = torch.from_numpy(local_file_paths).float()

        # move local file names to gpu
        local_file_paths = local_file_paths.cuda(non_blocking=True)
        #print('local_file_paths shape is ', local_file_paths.shape)





        # forward pass
        feats = model(images).clone()
        #print('feats shape is ', feats.shape)

        # getting attentional maps
        local_attentional_maps = model.get_last_selfattention(images).clone()
        bz = local_attentional_maps.shape[0] # batch size
        # we keep only the output patch attention
        local_attentional_maps = local_attentional_maps[:, :, 0, 1:].reshape(bz, -1)
        #print('attentions shape is ', local_attentional_maps.shape)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None and file_names is None and file_paths is None and attentional_maps is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            file_names = torch.zeros(len(data_loader.dataset), local_file_names.shape[-1])
            file_paths = torch.zeros(len(data_loader.dataset), local_file_paths.shape[-1])
            attentional_maps = torch.zeros(len(data_loader.dataset), local_attentional_maps.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
                file_names = file_names.cuda(non_blocking=True)
                file_paths = file_paths.cuda(non_blocking=True)
                attentional_maps = attentional_maps.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            print(f"Storing file names into tensor of shape {file_names.shape}")
            print(f"Storing file paths into tensor of shape {file_paths.shape}")
            print(f"Storing attentional maps into tensor of shape {attentional_maps.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)
        if args.inference_up_to:
            counter += (dist.get_world_size() * args.batch_size_per_gpu)
            cumulative_indexes[index_all] = True


        # share attentional maps between processes
        attentional_maps_all = torch.empty(
                dist.get_world_size(),
                local_attentional_maps.size(0),
                local_attentional_maps.size(1),
                dtype=local_attentional_maps.dtype,
                device=local_attentional_maps.device,
                )
        attentional_maps_output_l = list(attentional_maps_all.unbind(0))
        attentional_maps_output_all_reduce = torch.distributed.all_gather(attentional_maps_output_l, local_attentional_maps, async_op=True)
        attentional_maps_output_all_reduce.wait()

        # share file names between processes
        file_names_all = torch.empty(
                dist.get_world_size(),
                local_file_names.size(0),
                local_file_names.size(1),
                dtype=local_file_names.dtype,
                device=local_file_names.device,
                )
        file_names_output_l = list(file_names_all.unbind(0))
        file_names_output_all_reduce = torch.distributed.all_gather(file_names_output_l, local_file_names, async_op=True)
        file_names_output_all_reduce.wait()

        # share file paths between processes
        file_paths_all = torch.empty(
                dist.get_world_size(),
                local_file_paths.size(0),
                local_file_paths.size(1),
                dtype=local_file_paths.dtype,
                device=local_file_paths.device,
                )
        file_paths_output_l = list(file_paths_all.unbind(0))
        file_paths_output_all_reduce = torch.distributed.all_gather(file_paths_output_l, local_file_paths, async_op=True)
        file_paths_output_all_reduce.wait()

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                file_names.index_copy_(0, index_all, torch.cat(file_names_output_l))
                file_paths.index_copy_(0, index_all, torch.cat(file_paths_output_l))
                attentional_maps.index_copy_(0, index_all, torch.cat(attentional_maps_output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                file_names.index_copy_(0, index_all.cpu(), torch.cat(file_names_output_l).cpu())
                file_paths.index_copy_(0, index_all.cpu(), torch.cat(file_paths_output_l).cpu())
                attentional_maps.index_copy_(0, index_all.cpu(), torch.cat(attentional_maps_output_l).cpu())

    if args.inference_up_to:
        features = features[cumulative_indexes]
        file_names = file_names[cumulative_indexes]
        file_paths = file_paths[cumulative_indexes]
        attentional_maps = attentional_maps[cumulative_indexes]

    return features, file_names, file_paths, attentional_maps

class ReturnIndexDataset(ImageFolderWithPaths):
    def __getitem__(self, idx):
        img, lab, path = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx, path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference using pretrained weights')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int, help='Per-GPU batch-size')
    parser.add_argument("--image_size", default=(1024, 1024), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/sky_images/', type=str)
    parser.add_argument('--inference_up_to', default=None, type=int, help='Inference up to n samples from the complete dataset')
    parser.add_argument("--directory_depth", default=4, type=int, help="This is the depth with which you want to spedify the file paths.")
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        features = torch.load(os.path.join(args.load_features, "feat.pth"))
    else:
        # need to extract features !
        extract_feature_pipeline(args)
        #features = extract_feature_pipeline(args)

    #if utils.get_rank() == 0:
        #if args.use_cuda:
            #features = features.cuda()

    dist.barrier()
