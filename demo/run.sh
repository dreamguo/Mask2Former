# export QT_QPA_PLATFORM=offscreen
# --video-input ./output_downsampled.mp4  \
# --output /home/mqguo/data/Mirrorworld/building_data/BugisJunction/scenes/BugisJunction_B1_S1_ZW_20250214_1338/data/human_masks \

python demo.py --config-file ../configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
  --img-file-input /home/mqguo/data/Mirrorworld/building_data/BugisJunction/scenes/BugisJunction_B1_S1_ZW_20250214_1338/data/images_2 \
  --output /home/mqguo/data/Mirrorworld/building_data/BugisJunction/scenes/BugisJunction_B1_S1_ZW_20250214_1338/data/human_masks_ADE20K_Sem_2 \
  --opts MODEL.WEIGHTS ../ckpts/ADE20K/model_final_6b4a3a.pkl

# COCO
# model_final_94dc52.pkl: Mask2Former	R50 (Panoptic Segmentation)
# ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml

# model_final_9fd0ae.pkl: Mask2Former	Swin-T (Panoptic Segmentation)
# ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml

# model_final_54b88a.pkl: Mask2Former	Swin-B (IN21k) (Panoptic Segmentation)
# ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml

# model_final_f07440.pkl: Mask2Former	Swin-L (IN21k) (Panoptic Segmentation)
# ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml


# ADE20K
# model_final_6b4a3a.pkl: Mask2Former	Swin-L (IN21k)  (Semantic Segmentation)
# ../configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml

# model_final_e0c58e.pkl: Mask2Former	Swin-L (IN21k)  (Panoptic Segmentation)
# ../configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml
