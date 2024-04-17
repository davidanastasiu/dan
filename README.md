## DAN

We present `DAN`, the multivariate time series prediction model we detail in our paper, "Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting," which will be presented at AAAI 2024. If you make use of our code or data, please cite our paper.

```bibtex
@inproceedings{li2024learning,
  title={Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting},
  author={Li, Yanhong and Xu, Jack and Anastasiu, David},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={171--179},
  year={2024}
}
```
## Preliminaries

Experiments were executed in an Anaconda 3 environment with Python 3.8.8. The following will create an Anaconda environment and install the requisite packages for the project.

```bash
conda create -n DAN python=3.8.8
conda activate DAN
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
python -m pip install -r requirements.txt
```

## Files organizations

Download the datasets from [here](https://clp.engr.scu.edu/static/datasets/seed_datasets.zip) and upzip the files in the data_provider directory. In the ./data_provider/datasets directory, there should now be 4 stream sensor (file names end with _S_fixed.csv) and 4 rain sensor (file names end with _R_fixed.csv) datasets.

Use parameter `--val_size` to set the number of randomly sampled validation points which will be used in the training process. 
Use parameter `--train_volume` to set the number of randomly sampled validation points which will be used in the training process. 
Use parameter `--stack_types` to set the model stacks specified for this sensor. 
Use parameter `--oversampling` to set the kruskal statistics threshold. 
Use parameter `--event_focus_level` to set the percent sampled without satisfying the KW threshold. 
Use parameter `--r_shift` is the shift positions of rain hinter. Set it to 288 if no predicted rain data is provided. Otherwise, set to 0~288 according to the length of known forecasted rain data.

Refer to the annotations in `options.py` for other parameter settings.

## Training mode

run `main-Saratoga.ipynb` as an example with model.train()

## Inferencing mode

The inferencing has already been included after running run `main.ipynb` You can also run `main.ipynb` without model.train() for inferencing only.


