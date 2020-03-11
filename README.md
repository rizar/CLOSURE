# The code for CLOSURE paper

[CLOSURE: Assessing Systematic Generalization of CLEVR Models](https://arxiv.org/abs/1912.05783)

by Dzmitry Bahdanau, Harm de Vries, Timothy J. O'Donnell, Shikhar Murty, Philippe Beaudoin, Yoshua Bengio, Aaron Courville

This repository contains the original code that was used to obtain the reported results. See the NOTICE\_AND\_LICENSE file for licensing information.

## Setup

The recommended way is to use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

```
cd closure_code
conda env create
conda activate closure
pip install -r requirements.txt -e .
export NMN=`pwd`
```

## Data

Can be dowloaded from [here](https://zenodo.org/record/3634090#.Xjc3R3VKi90).

## Preprocessing

Apply the same preprocessing that is done for the original CLEVR data ([see here](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr)).

After preprocessing, you should have `<test>_features.h5` and `<test>_questions.h5` files for each part of CLOSURE on which you want to test models.

## Training on CLEVR 

To train the GT-Vector-NMN model from the paper, run this command:

```
bash <closure_code>/scripts/train/ee_film_clevr.sh --data_dir <data> --val_part val --checkpoint_path model.pt
```

Here, `<data>` should contain preprocessed CLEVR data in files `train_features.h5`, `train_questions.h5`, `val_features.h5` and `val_questions.h5`. 

You can monitor the model's online performance on CLOSURE tests as well. To do this, preprocess the data as discussed above and use extra `--val-part` arguments.

`scripts/train` also contains the training scripts for Tensor-NMN, MAC, FiLM and the program generator (PG).

## Evaluation

```
bash <closure_code>/run_model.py --execution_engine model.pt --data_dir <data> --part <closure_test> --output_h5 output.h5
```
