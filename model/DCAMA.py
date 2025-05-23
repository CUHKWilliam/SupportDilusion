from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from .base.swin_transformer import SwinTransformer
from model.base.transformer import MultiHeadedAttention, PositionalEncoding
import copy

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1), -1).contiguous()

def reshape(x, size):
    size1 = torch.tensor(x.size()).float().cuda()
    # x = torch.logical_not(x.cuda())
    yxs = torch.stack(torch.where(x), dim=-1)
    ratio = size[0] / size1[0]
    yxs2 = (yxs * ratio).long()
    x2 = torch.zeros((size[0], size[1])).float().cuda()
    return yxs2


class DCAMA(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize, use_sc=False, use_pruning=False):
        super(DCAMA, self).__init__()
        self.use_sc = use_sc
        self.use_pruning = use_pruning
        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize
        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
            self.last_feat_size = [12, 12]
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, use_sc=use_sc)
        
        ## TODO:
        
        self.scorer2 = nn.ModuleList()
        for layer_idx in range(len(self.nlayers)):
            layer_num = self.nlayers[layer_idx]
            for idx in range(layer_num):
                self.scorer2.append(
                    nn.Sequential(
                        nn.Conv2d(256 * 2 ** layer_idx, 256 * 2 ** layer_idx, 1, 1),
                        # nn.ReLU(),
                        # nn.InstanceNorm2d(256 * 2 ** layer_idx),
                        # nn.Conv2d(256 * 2 ** layer_idx, 256 * 2 ** layer_idx, 1, 1),
                    )
                )
        self.scorer1 = nn.Sequential(
            nn.Linear(sum(self.nlayers) - self.nlayers[0], 1)    
        )
       
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask, nshot, predict_score=False):
        n_support_feats = []
        with torch.no_grad():
            for k in range(nshot):
                support_feats_= self.extract_feats(support_img[:, k])
                support_feats = copy.deepcopy(support_feats_)
                del support_feats_
                torch.cuda.empty_cache()
                n_support_feats.append(support_feats)
            query_feats = self.extract_feats(query_img)
        

        logit_mask = self.model(query_feats, n_support_feats, support_mask.clone(), nshot=nshot)
        ## TODO:
        MAX_SHOTS = 1
        if len(n_support_feats) >= MAX_SHOTS:
            nshot = MAX_SHOTS   
            n_support_query_f = []
            n_simi = []
            for i in range(len(n_support_feats)):
                support_f = n_support_feats[i]
                support_query_f = []
                simi_l = []
                simi_sum = []
                for l in range(len(query_feats)):
                    if l < self.stack_ids[0]:
                        continue
                    elif l < self.stack_ids[1]:
                        DCAMA_blocks = self.model.DCAMA_blocks[0]
                        pe = self.model.pe[0]
                    elif l < self.stack_ids[2]:
                        DCAMA_blocks = self.model.DCAMA_blocks[1]
                        pe = self.model.pe[1]
                    else:
                        DCAMA_blocks = self.model.DCAMA_blocks[2]
                        pe = self.model.pe[2]
                    a_support_f = support_f[l].clone()
                    coords = reshape(support_mask[0, i], a_support_f.size()[-2:])
                    b, ch, w, h = a_support_f.size()
                    a_support_f = a_support_f.view(b, ch, -1)
                    a_support_f = DCAMA_blocks.linears[0](pe(a_support_f.permute(0, 2, 1))).permute(0, 2, 1)
                    a_support_f = a_support_f.view(b, ch, w, h)
                    a_support_f = self.scorer2[l](a_support_f)
                    a_support_f = a_support_f[:, :, coords[:, 0], coords[:, 1]].mean(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, 1, a_support_f.size(-2), a_support_f.size(-1)))
                    # a_support_f[:, :, coords_reverse[:, 0], coords_reverse[:, 1]] *= 0.
                    query_feat = query_feats[l].view(b, ch, -1)
                    query_feat = DCAMA_blocks.linears[0](pe(query_feat.permute(0, 2, 1))).permute(0, 2, 1)
                    query_feat = query_feat.view(b, ch, w, h)
                    query_feat = self.scorer2[l](query_feat)
                    simi = ((query_feat * a_support_f).sum(1)  / torch.norm(query_feat, dim=1) / torch.norm(a_support_f, dim=1))[0]
                    simi_sum.append(simi)
                    # simi = torch.norm(query_feats[l] - a_support_f, dim=1)[0]
                    if l == 6:
                        simi_map = simi.clone()
                    simi = simi.view(-1).mean()
                    simi_l.append(simi)
                # simi_l = self.scorer1(torch.stack(simi_l, dim=0).unsqueeze(0)).squeeze(0)[0]
                n_simi.append(torch.stack(simi_l, dim=0).mean())
             
            n_simi = torch.stack(n_simi, dim=0)
            args = n_simi.argsort(descending=True)[:MAX_SHOTS]
            support_mask = support_mask[:, args, :, :]
            # n_support_feats = [n_support_feats[arg] for arg in args]
            n_simis = n_simi[args].max()
        else:
            n_simis = torch.tensor(0.).float().cuda()
        return logit_mask, n_simis
    
    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)

            n_simis = []
            simi_map = None
            for i in range(len(n_support_feats)):
                support_f = n_support_feats[i]
                support_query_f = []
                simi_l = []
                for l in range(len(query_feats)):
                    if l < self.stack_ids[0]:
                        continue
                    elif l < self.stack_ids[1]:
                        DCAMA_blocks = self.model.DCAMA_blocks[0]
                        pe = self.model.pe[0]
                    elif l < self.stack_ids[2]:
                        DCAMA_blocks = self.model.DCAMA_blocks[1]
                        pe = self.model.pe[1]
                    else:
                        DCAMA_blocks = self.model.DCAMA_blocks[2]
                        pe = self.model.pe[2]
                    a_support_f = support_f[l].clone()
                    coords = reshape(support_masks[0, i], a_support_f.size()[-2:])
                    b, ch, w, h = a_support_f.size()
                    a_support_f = a_support_f.view(b, ch, -1)
                    a_support_f = DCAMA_blocks.linears[0](pe(a_support_f.permute(0, 2, 1))).permute(0, 2, 1)
                    a_support_f = a_support_f.view(b, ch, w, h)
                    a_support_f = a_support_f[:, :, coords[:, 0], coords[:, 1]].mean(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, 1, a_support_f.size(-2), a_support_f.size(-1)))
                    # a_support_f[:, :, coords_reverse[:, 0], coords_reverse[:, 1]] *= 0.
                    query_feat = query_feats[l].view(b, ch, -1)
                    query_feat = DCAMA_blocks.linears[0](pe(query_feat.permute(0, 2, 1))).permute(0, 2, 1)
                    query_feat = query_feat.view(b, ch, w, h)
                    simi = ((query_feat * a_support_f).sum(1)  / torch.norm(query_feat, dim=1) / torch.norm(a_support_f, dim=1))[0]
                    # simi = torch.norm(query_feats[l] - a_support_f, dim=1)[0]
                    if l == 13:
                        simi_map = simi.clone()
                    simi = simi.view(-1).max()
                    simi_l.append(simi)
                simi_l = torch.stack(simi_l, dim=0).mean()
                n_simis.append(simi_l)
            n_simis = torch.stack(n_simis, dim=0)
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(), nshot)
            
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
            
            ## TODO: retrieval V1 ##
            MAX_SHOTS = 10
            '''
            if len(n_support_feats) > MAX_SHOTS:
                nshot = MAX_SHOTS   
                n_support_query_f = []
                n_simis = []
                for i in range(len(n_support_feats)):
                    support_f = n_support_feats[i]
                    support_query_f = []
                    simi_l = []
                    simi_sum = []
                    for l in range(len(query_feats)):
                        if l < self.stack_ids[0]:
                            continue
                        elif l < self.stack_ids[1]:
                            DCAMA_blocks = self.model.DCAMA_blocks[0]
                            pe = self.model.pe[0]
                        elif l < self.stack_ids[2]:
                            DCAMA_blocks = self.model.DCAMA_blocks[1]
                            pe = self.model.pe[1]
                        else:
                            DCAMA_blocks = self.model.DCAMA_blocks[2]
                            pe = self.model.pe[2]
                        a_support_f = support_f[l].clone()
                        coords = reshape(support_masks[0, i], a_support_f.size()[-2:])
                        b, ch, w, h = a_support_f.size()
                        a_support_f = a_support_f.view(b, ch, -1)
                        a_support_f = DCAMA_blocks.linears[0](pe(a_support_f.permute(0, 2, 1))).permute(0, 2, 1)
                        a_support_f = a_support_f.view(b, ch, w, h)
                        a_support_f = a_support_f[:, :, coords[:, 0], coords[:, 1]].mean(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, 1, a_support_f.size(-2), a_support_f.size(-1)))
                        # a_support_f[:, :, coords_reverse[:, 0], coords_reverse[:, 1]] *= 0.
                        query_feat = query_feats[l].view(b, ch, -1)
                        query_feat = DCAMA_blocks.linears[0](pe(query_feat.permute(0, 2, 1))).permute(0, 2, 1)
                        query_feat = query_feat.view(b, ch, w, h)
                        simi = ((query_feat * a_support_f).sum(1)  / torch.norm(query_feat, dim=1) / torch.norm(a_support_f, dim=1))[0]
                        simi_sum.append(simi)
                        # simi = torch.norm(query_feats[l] - a_support_f, dim=1)[0]
                        if l == 6:
                            simi_map = simi.clone().detach().cpu().numpy()
                        simi = simi.view(-1).max()
                        simi_l.append(simi)
                    simi_l = torch.stack(simi_l, dim=0).mean()
                    n_simis.append(simi_l)
                n_simis = torch.stack(n_simis, dim=0)
                # nshot = max((n_simis > 0.).sum(), 1)
                nshot = len(n_simis)
                support_masks = support_masks[:, n_simis.argsort(descending=True)[:nshot], :, :]
                n_support_feats = [n_support_feats[i] for i in n_simis.argsort(descending=True)[:nshot]]
            else:
                n_simis = torch.tensor(0.).float().cuda()
                simi_map = None
            ## TODO: retriever V2
            '''
            '''
            MAX_SHOTS = 30
            if len(n_support_feats) > MAX_SHOTS:
                nshot = MAX_SHOTS
                n_support_query_f = []
                n_simis = []
                support_f_list = []
                n_support_feats2 = []
                query_feats2 = []
                for i in range(len(n_support_feats)):
                    support_f = n_support_feats[i]
                    n_support_feats2_l = []
                    query_feats2_l = []
                    for l in range(len(query_feats)):
                        if l < self.stack_ids[0]:
                            continue
                        elif l < self.stack_ids[1]:
                            DCAMA_blocks = self.model.DCAMA_blocks[0]
                            pe = self.model.pe[0]
                        elif l < self.stack_ids[2]:
                            DCAMA_blocks = self.model.DCAMA_blocks[1]
                            pe = self.model.pe[1]
                        else:
                            DCAMA_blocks = self.model.DCAMA_blocks[2]
                            pe = self.model.pe[2]
                        a_support_f = support_f[l].clone()
                        coords = reshape(support_masks[0, i], a_support_f.size()[-2:])
                        b, ch, w, h = a_support_f.size()
                        a_support_f = a_support_f.view(b, ch, -1)
                        a_support_f = DCAMA_blocks.linears[0](pe(a_support_f.permute(0, 2, 1))).permute(0, 2, 1)
                        a_support_f = a_support_f.view(b, ch, w, h)
                        a_support_f = a_support_f[:, :, coords[:, 0], coords[:, 1]].mean(-1)
                        n_support_feats2_l.append(a_support_f)
                        query_feat = query_feats[l].view(b, ch, -1)
                        query_feat = DCAMA_blocks.linears[0](pe(query_feat.permute(0, 2, 1))).permute(0, 2, 1)
                        query_feat = query_feat.view(b, ch, w, h)
                        query_feats2_l.append(query_feat)
                    n_support_feats2.append(n_support_feats2_l)
                    query_feats2.append(query_feats2_l)
                n_support_feats3 = [[] for _ in range(len(query_feats2[0]))]
                selected = []
                for i in range(MAX_SHOTS):
                    simi_min = -100
                    idx_min = -1
                    for idx in range(len(n_support_feats2)):
                        if idx in selected:
                            continue
                        support_feats2 = n_support_feats2[idx]
                        simi = []
                        for l in range(len(query_feats2[i])):
                            support_feats_avg = torch.stack(n_support_feats3[l] + [support_feats2[l]], dim=0).mean(0)
                            query_feat = query_feats2[i][l]
                            a_support_f = support_feats_avg.unsqueeze(-1).unsqueeze(-1).repeat(
                                (1, 1, query_feat.size(-2), query_feat.size(-1)))
                            simi_l = ((query_feat * a_support_f).sum(1) / torch.norm(query_feat, dim=1) / torch.norm(
                                a_support_f, dim=1))[0].view(-1).max()
                            simi.append(simi_l)
                        simi = torch.stack(simi, dim=0).mean()
                        if simi > simi_min:
                            simi_min = simi
                            idx_min = idx
                    support_feats2_argmin = n_support_feats2[idx]
                    for l2 in range(len(query_feats2[0])):
                        n_support_feats3[l2].append(n_support_feats2[idx_min][l2])
                    selected.append(idx_min)
                n_support_feats4 = []
                for idx in selected:
                    n_support_feats4.append(n_support_feats[idx])
                support_masks = support_masks[:, torch.tensor(selected).long().cuda(), :, :]
                n_support_feats = n_support_feats4
                simi_map = None
            else:
                n_simis = torch.tensor(0.).float().cuda()
                simi_map = None
            '''
            ## TODO: v3
            
            MAX_SHOTS = 30
            if len(n_support_feats) > MAX_SHOTS and self.use_pruning:
                nshot = MAX_SHOTS
                n_support_query_f = []
                n_simis = []
                support_f_list = []
                n_support_feats2 = []
                query_feats2 = []
                for i in range(len(n_support_feats)):
                    support_f = n_support_feats[i]
                    n_support_feats2_l = []
                    query_feats2_l = []
                    for l in range(len(query_feats)):
                        if l < self.stack_ids[0]:
                            continue
                        elif l < self.stack_ids[1]:
                            DCAMA_blocks = self.model.DCAMA_blocks[0]
                            pe = self.model.pe[0]
                        elif l < self.stack_ids[2]:
                            DCAMA_blocks = self.model.DCAMA_blocks[1]
                            pe = self.model.pe[1]
                        else:
                            DCAMA_blocks = self.model.DCAMA_blocks[2]
                            pe = self.model.pe[2]
                        a_support_f = support_f[l].clone()
                        coords = reshape(support_masks[0, i], a_support_f.size()[-2:])
                        b, ch, w, h = a_support_f.size()
                        a_support_f = a_support_f.view(b, ch, -1)
                        a_support_f_tmp = DCAMA_blocks.linears[0](pe(a_support_f.permute(0, 2, 1))).permute(0, 2, 1)
                        a_support_f = a_support_f_tmp / a_support_f_tmp.norm(dim=1, keepdim=True) * DCAMA_blocks.linears[1](pe(a_support_f.permute(0, 2, 1))).permute(0, 2, 1)
                        a_support_f = a_support_f.view(b, ch, w, h)
                        a_support_f = a_support_f[:, :, coords[:, 0], coords[:, 1]].mean(-1)
                        n_support_feats2_l.append(a_support_f)
                        query_feat = query_feats[l].view(b, ch, -1)
                        query_feat_tmp = DCAMA_blocks.linears[0](pe(query_feat.permute(0, 2, 1))).permute(0, 2, 1)
                        query_feat = query_feat_tmp / query_feat_tmp.norm(dim=1, keepdim=True) * DCAMA_blocks.linears[1](pe(query_feat.permute(0, 2, 1))).permute(0, 2, 1)
                        query_feat = query_feat.view(b, ch, w, h)
                        query_feats2_l.append(query_feat)
                    n_support_feats2.append(n_support_feats2_l)
                    query_feats2.append(query_feats2_l)
                n_support_feats3 = [[] for _ in range(len(query_feats2[0]))]
                selected = []
                for i in range(MAX_SHOTS):
                    simi_min = -100
                    idx_min = -1
                    for idx in range(len(n_support_feats2)):
                        if idx in selected:
                            continue
                        support_feats2 = n_support_feats2[idx]
                        simi = []
                        for l in range(len(query_feats2[i])):
                            support_feats_avg = torch.stack(n_support_feats3[l] + [support_feats2[l]], dim=0).mean(0)
                            query_feat = query_feats2[i][l]
                            a_support_f = support_feats_avg.unsqueeze(-1).unsqueeze(-1).repeat(
                                (1, 1, query_feat.size(-2), query_feat.size(-1)))
                            simi_l = ((query_feat * a_support_f).sum(1))[0].view(-1).max()
                            simi.append(simi_l)
                        simi = torch.stack(simi, dim=0).mean()
                        if simi > simi_min:
                            simi_min = simi
                            idx_min = idx
                    support_feats2_argmin = n_support_feats2[idx]
                    for l2 in range(len(query_feats2[0])):
                        n_support_feats3[l2].append(n_support_feats2[idx_min][l2])
                    selected.append(idx_min)
                n_support_feats4 = []
                for idx in selected:
                    n_support_feats4.append(n_support_feats[idx])
                support_masks = support_masks[:, torch.tensor(selected).long().cuda(), :, :]
                n_support_feats = n_support_feats4
                simi_map = None
            else:
                n_simis = torch.tensor(0.).float().cuda()
                simi_map = None
            
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(), nshot)

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)
    
        return logit_mask.argmax(dim=1), n_simis, simi_map

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids, use_sc=False):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids

        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5, use_sc=use_sc))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        self.mixer1 = nn.Sequential(nn.Conv2d(outch3+2*in_channels[1]+2*in_channels[0], outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))
    
        

    def forward(self, query_feats, support_feats, support_mask, nshot=1):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            # if nshot == 1:
            #     support_feat = support_feats[idx]
            #     mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
            #                          align_corners=True).view(support_feat.size()[0], -1)
            #     support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            # else:
            support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
            support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
            mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
            mask = mask.view(bsz, -1)

            # DCAMA blocks forward
            DCAMA_blocks = None
            pe = None
            if idx < self.stack_ids[1]:
                DCAMA_blocks = self.DCAMA_blocks[0]
                pe = self.pe[0]
            elif idx < self.stack_ids[2]:
                DCAMA_blocks = self.DCAMA_blocks[1]
                pe = self.pe[1]
            else:
                DCAMA_blocks = self.DCAMA_blocks[2]
                pe = self.pe[2]
            coarse_mask = DCAMA_blocks(pe(query), pe(support_feat), mask)
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))


        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        mix = coarse_masks1 + coarse_masks2
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3
        mix = self.conv5(mix)

        # skip connect 1/8 and 1/4 features (concatenation)
        # if nshot == 1:
        #     support_feat = support_feats[self.stack_ids[1] - 1]
        # else:
        support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
        # if nshot == 1:
        #     support_feat = support_feats[self.stack_ids[0] - 1]
        # else:
        support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)

        # mixer blocks forward
        out = self.mixer1(mix)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        return logit_mask

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)
