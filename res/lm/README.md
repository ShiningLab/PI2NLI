# PI2NLI - Language Models

## Directory
The language models directory contains the following model folders:

```
res/lm
├── README.md
├── roberta-large
├── roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
├── xlnet-large-cased
└── xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli
```

## Downloads
You can download the models from HuggingFace as follows:
```sh
# Make sure you have git-lfs installed (https://git-lfs.com)
$ git lfs install

# Clone the repositories
$ git clone https://huggingface.co/FacebookAI/roberta-large
$ git clone https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
$ git clone https://huggingface.co/xlnet/xlnet-large-cased
$ git clone https://huggingface.co/ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```bibtex
@article{liu2019roberta,
  title={Roberta: A robustly optimized bert pretraining approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@inproceedings{NEURIPS2019_dc6a7e65,
 author = {Yang, Zhilin and Dai, Zihang and Yang, Yiming and Carbonell, Jaime and Salakhutdinov, Russ R and Le, Quoc V},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {XLNet: Generalized Autoregressive Pretraining for Language Understanding},
 url = {https://proceedings.neurips.cc/paper_files/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf},
 volume = {32},
 year = {2019}
}

@inproceedings{nie-etal-2020-adversarial,
    title = "Adversarial {NLI}: A New Benchmark for Natural Language Understanding",
    author = "Nie, Yixin  and
      Williams, Adina  and
      Dinan, Emily  and
      Bansal, Mohit  and
      Weston, Jason  and
      Kiela, Douwe",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    Xpublisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.441",
    doi = "10.18653/v1/2020.acl-main.441",
    pages = "4885--4901",
    abstract = "We introduce a new large-scale NLI benchmark dataset, collected via an iterative, adversarial human-and-model-in-the-loop procedure. We show that training models on this new dataset leads to state-of-the-art performance on a variety of popular NLI benchmarks, while posing a more difficult challenge with its new test set. Our analysis sheds light on the shortcomings of current state-of-the-art models, and shows that non-expert annotators are successful at finding their weaknesses. The data collection method can be applied in a never-ending learning scenario, becoming a moving target for NLU, rather than a static benchmark that will quickly saturate.",
}
```