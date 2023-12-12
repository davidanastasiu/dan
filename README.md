## DAN

We present `DAN`, the multivariate time series prediction model we detail in our paper, "Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting," which will be presented at AAAI 2024. If you make use of our code or data, please cite our paper.

```bibtex
@inproceedings{Li2023AAAI23,
  author    = {Yanhong Li and David C. Anastasiu},
  title     = {Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting},
  booktitle = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI} 2024},
  pages     = {},
  publisher = {{AAAI} Press},
  year      = {2024},
}
```

## Preliminaries

Experiments were executed in an Anaconda 3 environment with Python 3.8.3. The following will create an Anaconda environment and install the requisite packages for the project.

```bash
conda create --name dan python=3.8.3
conda activate dan
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 -c pytorch
python -m pip install -r requirements.txt
```

## File organization

Download the datasets from [here](https://clp.engr.scu.edu/static/datasets/seed_datasets.zip) and upzip the files in the data_provider directory. In the ./data_provider/datasets directory, there should now be 4 stream sensor (file names end with _S_fixed.csv) and 4 rain sensor (file names end with _R_fixed.csv) datasets.

Use parameter `--val_size` to set the number of randomly sampled validation points which will be used in the training process. 
Use parameter `--train_volume` to set the number of randomly sampled validation points which will be used in the training process. 
Use parameter `--stack_types` to set the model stacks specified for this sensor. 
Use parameter `--oversampling` to set the kruskal statistics threshold. 
Use parameter `--event_focus_level` to set the percent sampled without satisfying the KW threshold. 
Use parameter `--r_shift` to set the shift positions of rain hinter. Set it to 288 if no predicted rain data is provided. Otherwise, set to 0~288 according to the length of known forecasted rain data.

Refer to the annotations in `options.py` for other parameter setting.


## Training mode

run `main.ipynb` with model.train()

## Inference mode

The inferencing has already been included after running run 'main.ipynb'. You can also run 'main.ipynb' without model.train() for inferencing only.

