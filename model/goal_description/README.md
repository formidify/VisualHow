# Problem Description Generation

This code implements the Problem Description Generation model.

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

