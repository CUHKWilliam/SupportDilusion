python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=18024 \
./train_retriever.py --datapath "/home/bkdongxianchi/MY_MOT/TWL/data" \
           --benchmark coco \
           --fold 0 \
           --bsz 1 \
           --nworker 8 \
           --backbone resnet50 \
           --feature_extractor_path "/home/bkdongxianchi/MY_MOT/TWL/DCAMA/backbones/resnet50_a1h-35c100f8.pth" \
           --logpath "/home/bkdongxianchi/MY_MOT/TWL/DCAMA/log" \
           --lr 1e-4 \
           --nepoch  50 \
           --load "/home/bkdongxianchi/MY_MOT/TWL/DCAMA/checkpoint/coco-20i/resnet50_fold0.pt" \
           --nshot 1
#           --load "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/checkpoint/coco-20i/resnet50_fold0.pt" \
