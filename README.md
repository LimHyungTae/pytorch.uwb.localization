# Pytorch UWB Localization 

Official page of [RONet](https://ieeexplore.ieee.org/abstract/document/8968551), which is published @IROS'19

Since the original code is based on Tensorflow, now I've port the original algorithm to pytorch.

![before](/materials/test.gif)


## ToDo
- [ ] Port RONet
- [ ] Port Bi-LSTM
- [x] Set training pipeline
- [x] Visualize training procedure
- [x] Autosave the best model

## Environments

Please refer to `requirements.txt`

## Descriptions

### What is the UWB sensor?

UWB is abbv. for *Ultra-wideband*, and the sensor outputs only 1D range data.

![validation](/materials/validation.png)

More explanations are provided in [this paper](https://ieeexplore.ieee.org/abstract/document/8768568).

In summary, UWB data are likely to be vulnerable to noise, multipath problem, and so forth.

Thus, we leverage the nonliearity of deep learning to tackle that issue.

### Data

All data are contained in `uwb_dataset` and total eight sensors are deployed, whose positions are as follows:

![idnames](/materials/id_names.png)

Note that our experiment was conducted on **real-world** data by using [Pozyx UWB sensors](https://www.pozyx.io/?ppc_keyword=pozyx&gclid=CjwKCAiAm-2BBhANEiwAe7eyFHFbVb7B_eub3dTe9oIUqgN1XI6c9O4N8aOj6L24fZyAHMKQLRahQxoCqdgQAvD_BwE) and motion capture system.

(Please kindly keep in mint that Pozyx systems does not give precise range data :()


## Training

The training scripts come with several options, which can be listed with the `--help` flag. 
```bash
python3 main.py --help
```

The point is that it only takes a few minutes because the data of UWB are lightweight and simple! :)

Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python3 main.py --resume [path_to_previous_model]
```

## Validation

```bash
python3 main.py --evaluate [path_to_trained_model]
```


## Benchmark

	| Methods   |  RMSE |  Mean | Median | Variance | Error Max |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| RNN | 0.442 | 0.104 | 87.8 | 96.4 | 98.9 |
	| GRU | 0.351 | 0.078 | 92.8 | 98.4 | 99.6 |
	| LSTM | 0.281 | 0.059 | 95.5 | 99.0 | 99.7 |
	| Ours-200| **0.230** | **0.044** | **97.1** | **99.4** | **99.8** |

## Citation


- If you use our code or method in your work, please consider citing the following::

```
@INPROCEEDINGS {lim2019ronet,
  author = {Lim, Hyungtae and Park, Changgue and Myung, Hyun},
  title = {Ronet: Real-time range-only indoor localization via stacked bidirectional lstm with residual attention},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3241--3247},
  year = { 2019 },
  organization={IEEE}
}
@INPROCEEDINGS {lim2018stackbilstm,
  author = {Lim, Hyungtae and Myung, Hyun},
  title = {Effective Indoor Robot Localization by Stacked Bidirectional LSTM Using Beacon-Based Range Measurements},
  booktitle = {International Conference on Robot Intelligence Technology and Applications},
  pages={144--151},
  year = { 2018 }
  organization={Springer}
}

```

## Contact

Contact: Hyungtae Lim (shapelim@kaist.ac.kr)

Please create a new issue for code-related questions. Pull requests are welcome.
