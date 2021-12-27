# m5-forecasting-accuracy

## Environment Management

Dependencies are managed by [Pipenv](https://docs.pipenv.org/)

### Install dependencies

* With dev dependencies
```bash
$ pipenv install
```

### Activate Environment

```bash
$ pipenv shell
```

## Jobs Execution

### Dataset Maker
```bash
python -m src.jobs.dataset_maker \
    --raw-data-dir assets/data/raw \
    --export-dir assets/data/data_CA_1 \
    --main-file sales_train_evaluation.csv \
    --store-id CA_1
```

### Training Auto Arima
```bash
python -m src.jobs.autoarima_trainer \
    --data-dir assets/data/data_CA_1/dataset.csv \
    --assets-dir assets/data/data_CA_1
```