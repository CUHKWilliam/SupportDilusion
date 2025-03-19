r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import torch.nn.functional as F

average_loss = torch.tensor(0.).float().cuda()
global_idx = 0
def train(epoch, model, dataloader, optimizer, training):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    
    global average_loss, global_idx
    average_loss = average_loss.to("cuda:{}".format(torch.cuda.current_device()))
    stats = [[], []]
    criterion_score = nn.BCEWithLogitsLoss()
    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        logit_mask, score_preds = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], nshot=batch['support_imgs'].size(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        # loss_obj = loss.detach()
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        iou = area_inter[1] / area_union[1]
        
        if iou > 0.7 or iou  < 0.1:
            '''
            if iou < 0.1:
                img = batch['query_img'][0].permute(1, 2, 0).detach().cpu().numpy()
                img = img - img.min()
                img = img / img.max()
                cv2.imwrite('query_image.png', (img * 255).astype(np.uint8))
                img = batch['support_imgs'][0][0].permute(1, 2, 0).detach().cpu().numpy()
                img = img - img.min()
                img = img / img.max()
                cv2.imwrite('support_image.png', (img * 255).astype(np.uint8))
                cv2.imwrite('query_mask.png', (batch['query_mask'][0] * 255).detach().cpu().numpy().astype(np.uint8))
                cv2.imwrite('pred_mask.png', (pred_mask[0] * 255).detach().cpu().numpy().astype(np.uint8))
                cv2.imwrite('support_mask.png', (batch['support_masks'][0][0] * 255).detach().cpu().numpy().astype(np.uint8))
            '''
            if iou > 0.7:
                iou = torch.tensor(1.).float().cuda()
            else:
                iou = torch.tensor(0.).float().cuda()
            score_loss = criterion_score(score_preds, iou)
            stats[0].append(score_preds.detach().cpu().numpy())
            stats[1].append((area_inter[1] / area_union[1]).detach().cpu().numpy())
            print(score_preds, (area_inter[1] / area_union[1]))
            loss = score_loss

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    import matplotlib.pyplot as plt
    idx = 0
    plt.scatter(stats[0], stats[1], c="red", s=2, alpha=0.1)
    plt.savefig('stat.png')
    plt.close()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # Model initialization
    model = DCAMA(args.backbone, args.feature_extractor_path, False)
    device = torch.device("cuda", args.local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    

    params = model.state_dict()
    state_dict = torch.load(args.load)
    state_dict2 = {}
    for k in state_dict.keys():
        if "scorer" in k:
            continue
        state_dict2[k] = state_dict[k]
    state_dict = state_dict2
    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)
    
    model.load_state_dict(state_dict, strict=False)
    
    ## TODO:
    for i in range(len(model.module.model.DCAMA_blocks)):
        torch.nn.init.constant_(model.module.model.DCAMA_blocks[i].linears[1].weight, 0.)
        torch.nn.init.constant_(model.module.model.DCAMA_blocks[i].linears[1].bias, 1.)


    # Helper classes (for training) initialization
    optimizer = optim.SGD([{"params": model.module.model.parameters(), "lr": args.lr,
                            "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    Evaluator.initialize()
    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', args.nshot)
    if args.local_rank == 0:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.nshot)

    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.nepoch):
        dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)

        # evaluation
        if args.local_rank == 0:
            # with torch.no_grad():
            #     val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

            # Save the best model
            # if val_miou > best_val_miou:
            #     best_val_miou = val_miou
            #     Logger.save_model_miou(model, epoch, val_miou)
            Logger.save_model_miou(model, epoch , 1.)

            # Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            # Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            # Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            # Logger.tbd_writer.flush()

    if args.local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
