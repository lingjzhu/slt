Standalone extraction of the `ssvp_slt` MAE pretraining path.

Run with:

```bash
PYTHONPATH=/home/slimelab/Projects/slt/src \
python -m mae_pretraining.run_pretraining \
  data.base_data_dir=/path/to/data \
  data.dataset_names=dataset_a,dataset_b
```

This package intentionally excludes unrelated translation, CLIP, and fairseq-based code.
