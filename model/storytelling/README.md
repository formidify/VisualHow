# Solution Captions Generation

This code implements the Solution Captions Generation model.

Training your own network
------------------

You can run the following command for cross entropy loss

```bash
$ python train.py --id XE --data_dir DATADIR --start_rl -1
```

Then you can run the following command for AREL model

```bash
$ python train_AREL.py --id AREL --start_from_model PRETRAINED_MODEL
```

Evaluating your own network
------------------

You can run the following command

```bash
$ python train.py --option test --beam_size 3 --start_from_model data/save/XE/model.pth
```

