## Unbiased Pairwise Learning from Implicit Feedback for Recommender Systems without Biased Variance Control
This repository contains the experiments related with the paper "**Unbiased Pairwise Learning from Implicit Feedback for Recommender Systems without Biased Variance Control**", which has been accepted by [SigIR'23](https://sigir.org/sigir2023/).

The implementation is based on the original [code](https://github.com/usaito/unbiased-pairwise-rec) from [Yuta Saito](https://usaito.github.io/) for the paper of "**Unbiased Pairwise Learning from Biased Implicit Feedback**". We added the implementation of UPL and MF-DU based on the work of usaito.

If you find the code is helpful, please cite our paper.
```
@inproceedings{ren2023upl,
  title={Unbiased Pairwise Learning from Implicit Feedback for Recommender Systems without Biased Variance Control},
  author={Yi Ren and Hongyan Tang and Jiangpeng Rong and Siwen Zhu},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023},
}
```
### Running the code

First, to preprocess the datasets, navigate to the `src/` directory and run the command

```bash
python preprocess_datasets.py -d coat yahoo
```

Then, run the following command in the same directory

```bash
for data in yahoo coat
  do
  for model in wmf relmf bpr ubpr upl_bpr relmf_du ubpr_nclip
  do
    python main.py -m $model -d $data -r 10
  done
done
```

After running the experimens, you can summarize the results by running the following command in the `src/` directory.

```bash
python summarize_results.py -d yahoo coat
```

Once the code is finished executing, you can find the summarized results in `./paper_results/` directory.

### Acknowledgement

We thank [Yuta Saito](https://usaito.github.io/) for the helpful code. 
