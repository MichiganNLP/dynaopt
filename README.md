# DynaOpt: Dynamic Reward Adjustment in Multi-Reward Reinforcement Learning for Counselor Reflection Generation

Code for ``Dynamic Reward Adjustment in Multi-Reward Reinforcement Learning for Counselor Reflection Generation``

## Quickstart
### 1. Prerequisites

Clone the repository:
```bash
git clone https://github.com/mindojune/dynaopt.git
cd dynaopt
```

Next, create a new environment and install the required packages:
```bash
conda create -n dynaopt python=3.9
conda activate bolt
pip install -r requirements
```


### 2. Download the pretrained weight for the reflection scorer.
Download the weight [here](https://drive.google.com/file/d/1RPvMVLe7WS_spOvQI8FmPz6khI-MWWtA/view?usp=drive_link).
Put the weight inside `dynaopt/weights`.

### 3. Running the models

First, train the warm-start model in a supervised manner.

```bash
python supervised_train.py --experiment MI --num_epochs 5
```

Save the path of your trained model and run the inference step over test data.
```bash
start_dir={your supervised model}
python test_util.py --experiment MI_rl --model_start_dir $start_dir
```

Next, you can train the rl models. We use the k-self critical sequencec training algorithm in this project.
Refer to the following table to train differnet models.
| model  | script / flag  |
|---|---|
| Round | rl_train.py / round  |
| Uniform Weighted  | rl_train.py / weighted  |
|  [DORB](https://aclanthology.org/2020.emnlp-main.625/) | rl_train.py / bandit  |
| DynaOpt (Ours)  | rl_train.py / bandit_weighted  |
|  C-Dynaopt (Ours) | con_rl_train.py  / None  |

```bash
python rl_train.py --learning_mode bandit_weighted --seed $i --experiment MI_rl --model_start_dir $start_dir
python con_rl_train.py --seed $i  --experiment MI_rl --model_start_dir $start_dir
python rl_train.py --learning_mode weighted --seed $i --experiment MI_rl --model_start_dir $start_dir
python rl_train.py --learning_mode round --seed $i  --experiment MI_rl --model_start_dir $start_dir
python rl_train.py --learning_mode bandit --seed $i  --experiment MI_rl --model_start_dir $start_dir
```

And compute the statistics on the test data.
```bash
python compute_stats.py --dir ./outputs
```



### License
Our project is licensed under the Apache License 2.0, ensuring open access and collaboration, with due credit given to the original work which forms the backbone of our codebase.

### This code makes use of the [Keep it Simple code by Laban et al.](https://github.com/tingofurro/keep_it_simple/)
Specifically, we use the code for the k self-critical sequence training algorithm..
```bibtex
@inproceedings{laban2021keep_it_simple,
  title={Keep It Simple: Unsupervised Simplification of Multi-Paragraph Text},
  author={Philippe Laban and Tobias Schnabel and Paul N. Bennett and Marti A. Hearst},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  volume={1},
  year={2021}
}
```


### Cite the work

If you make use of the code, models, or algorithm, please cite our paper:
```bibtex
@inproceedings{laban2021keep_it_simple,
  title={Dynamic Reward Adjustment in Multi-Reward Reinforcement Learning for Counselor Reflection Generation},
    author = "Min, Do June  and
    P{\'e}rez-Rosas, Ver{\'o}nica  and
    Resnicow, Kenneth  and
    Mihalcea, Rada",
  booktitle={COLING 2024},
  volume={},
  year={2024}
}
```
