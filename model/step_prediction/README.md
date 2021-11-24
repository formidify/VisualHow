# Solution Steps Prediction

This code implements the Solution Steps Prediction model.

Training your own network
------------------

You can run the following command

```bash
$ sh train_grid_wikihow_att_supervised_ce.sh
```

Evaluating your own network
------------------

You can run the following command

```bash
$ CUDA_VISIBLE_DEVICES=0 python eval.py --data_path {data_path}
```

