# PCIE_Ego4D_Social_Interaction

This repo presents our team's PCIE_Interaction solution for the Ego4D Social Interaction Challenge at CVPR 2025, addressing both Looking At Me (LAM) and Talking To Me (TTM) tasks. The challenge requires accurate detection of social interactions between subjects and the camera wearer, with LAM relying exclusively on face crop sequences and TTM combining speaker face crops with synchronized audio segments. In the LAM track, we employ face quality enhancement and ensemble methods. For the TTM task, we extend visual interaction analysis by fusing audio and visual cues, weighted by a visual quality score. Our approach achieved a mean average precision (mAP) of 0.81 and 0.71 on the LAM and TTM challenge leaderboards.

Our approach for social interaction analysis extends the LAM to the TTM task by employing separate models for visual and audio modalities. Based on the dataset characteristics, we introduce visual and audio filters before fusing the scores from both models, yielding a robust final interaction score for TTM.

# Looking At Me

Please refer to the GitHub repository ([GitHub repo](https://github.com/EGO4D/social-interactions/tree/lam)) for instructions on data preparation and baseline model generation.

Run the following script to perform inference on the face crop test dataset:
```bash
python run.py --eval --checkpoint ${checkpoint_path} --exp_path ${eval_output_dir} --infer --test_path ${test_dataset_path}
```
The results of Looking at Me will be in .csv format


Run the following script to generate results for the Talking to me task on the .csv file: 
```bash
python csv2json_ttm.py --input ${lam_results.csv} --output ${ouput.json} 
```

## Citation 
```
@article{Lertniphonphan2025PCIE\_InteractionSF,
  title={PCIE\_Interaction Solution for Ego4D Social Interaction Challenge},
  author={Kanokphan Lertniphonphan and Feng Chen and Junda Xu and Fengbu Lan and Jun Xie and Tao Zhang and Zhepeng Wang},
  journal={ArXiv},
  year={2025},
}
```

## References
Our LAM is based on Ego4d Social Benchmark ([GitHub repo](https://github.com/EGO4D/social-interactions/tree/lam)).
For audio processing, we follow Ego4D-QuAVF-TTM-CVPR23 [GitHub repo](https://github.com/hsi-che-lin/Ego4D-QuAVF-TTM-CVPR23/tree/b6a866f8dcaf07d8fd5af800d2ca1c3e4fec544c).
