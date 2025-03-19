python ./test.py --datapath "/research/d4/gds/wltang21/data" \
                 --benchmark coco \
                 --fold 0 \
                 --bsz 1 \
                 --nworker 1 \
                 --backbone resnet101 \
                 --feature_extractor_path "/research/d4/gds/wltang21/logistic_project/DCAMA/backbones/resnet101_a1h-36d3f2aa.pth" \
                 --logpath "./logs" \
		 --load "/research/d4/gds/wltang21/logistic_project/DCAMA/log/train/fold_0_ft_v0_res-101/model_40.pt" \
                 --nshot 1 \
                 --use_sc \
                 --use_pruning
		 # --load "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/checkpoint/coco-20i/resnet50_fold0.pt" \
#                 --visualize
#checkpoint/coco-20i/resnet50_fold0.pt
# log/train/fold_0_ft_v0/best_model.pt
