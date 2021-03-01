# Pytorch UWB Localization 

Official page of RONet, which is published @IROS'19

Since the original code is based on Tensorflow, now I've port the original algorithm to pytorch.

![before](/materials/test.gif){: .center-block :}

[test](.requirements.txt)

## ToDo
- [ ] Port RONet
- [ ] Port Bi-LSTM
- [x] Set training pipeline
- [x] Visualize training procedure
- [x] Autosave the best model

## Descriptions

[여기](https://github.com/IntelRealSense/librealsense/wiki/D400-Series-Visual-Presets)


## Training
The training scripts come with several options, which can be listed with the `--help` flag. 
```bash
python3 main.py --help
```

Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python3 main.py --resume [path_to_previous_model]
```

## Validation

```bash
python3 main.py --evaluate [path_to_trained_model]
```


## Check

plot_rmse

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
