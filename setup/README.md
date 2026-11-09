# Server setup

Scripts used once on the university GPU server, run from the repository root:

| File | What it does |
| --- | --- |
| `setup_and_prepare.sh` | finds the downloaded data, creates the Python environment, installs the packages, checks the GPU, prepares the data (`data/prepare_data.py`) and checks the balance (`train/balance.py`) |
| `setup_log.txt` | the output of that run (600 places, GPU check, data preparation, balance report) |
| `start_training.sh` | starts all training runs (`train/run_experiments.sh`) in the background on GPU 0 |

```bash
bash setup/setup_and_prepare.sh
bash setup/start_training.sh
tail -f train/train_log.txt
```
