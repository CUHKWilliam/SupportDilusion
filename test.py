r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import cv2
import numpy as np
import os
# from gpu_mem_track import MemTracker

from calflops import calculate_flops


def test(model, dataloader, nshot):
    r""" Test """
	
    input_shape = (1, 3, 384, 384)
    param_cnt = 0
    print(sum(p.numel() for p in model.parameters()))
    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        nshot = batch['support_imgs'].size(1)
        ## TODO:
        batch = utils.to_cuda(batch)
        # gpu_tracker.track()
        pred_mask, simi, simi_map = model.module.predict_mask_nshot(batch, nshot=nshot)
        # gpu_tracker.track()
        torch.cuda.synchronize()
        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        
        ## TODO:
        iou = area_inter[1] / area_union[1]
        
        '''
        cv2.imwrite('debug/query.png', cv2.imread("/home/bkdongxianchi/MY_MOT/TWL/data/COCO2014/{}".format(batch['query_name'][0])))
        cv2.imwrite('debug/query_mask.png', (batch['query_mask'][0] * 255).detach().cpu().numpy().astype(np.uint8))
        cv2.imwrite('debug/support_{:.3}.png'.format(iou.item()), cv2.imread('/home/bkdongxianchi/MY_MOT/TWL/data/COCO2014/{}'.format(batch['support_names'][0][0])))
        cv2.imwrite('debug/support_mask_{:.3}.png'.format(iou.item()), (batch['support_masks'][0][0] * 255).detach().cpu().numpy().astype(np.uint8))
        simi_map = simi_map - simi_map.min()
        simi_map = (simi_map / simi_map.max() * 255).detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite('debug/simi_map_{:.3}.png'.format(iou.item()), simi_map)
        
        if os.path.exists('debug/stats.txt'):
            with open('debug/stats.txt', "a") as f:
                f.write("{} {}\n".format(simi.item(), iou.item()))
        else:
            with open('debug/stats.txt', 'w') as f:
                f.write('{} {}\n'.format(simi.item(), iou.item()))
        '''

        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  iou_b=area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    Logger.initialize(args, training=False)

    # Model initialization
    model = DCAMA(args.backbone, args.feature_extractor_path, args.use_original_imgsize)
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    params = model.state_dict()
    state_dict = torch.load(args.load)
    
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    state_dict2 = {}
    for k, v in state_dict.items():
        if 'scorer' not in k:
            state_dict2[k] = v
    state_dict = state_dict2

    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)
   
 
    try:
        model.load_state_dict(state_dict, strict=True)
    except:
        for k in params.keys():
            if k not in state_dict.keys():
                state_dict[k] = params[k]
        model.load_state_dict(state_dict, strict=True)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
