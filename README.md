# Moire Removal via Attentive Fractal Network

create and activate a virtual environment with Python >= 3.6

Install requirements

```bash
pip install -r requirements.txt
```

## Training

There are a few code you must modify before training.

1. Modify variable `TR_INPUT` and `TR_GT` in `data.py`.
   TR_INPUT indicates the input path for the whole released training set
   TR_GT indicates the gt path for the whole released training set

2. (optional) Dump a numpy.ndarray for indexing train set and val set with ndim==1 into train.pkl and val.pkl

3. Run command below to start training.

   ```bash
    python main.py -M AFN --batch_size <BSZ> --lr <initial value used in cos lr> --max_epochs <n_epoch>
   ```

You may want to check full command line arguments available with `--help`.

## Inferring

1. Set `TS_INPUT` in `data.py` to the path of result for testing.

2. Run command below to start inferring.

   ```bash
   python main.py --mode test -M AFN -C <checkpoint path> --timestamp final_resu_test
   ```

   Results will be located in `res/final_resu_test`

## Acknowledgement

This repo is still under construction.

By OIerM team.
