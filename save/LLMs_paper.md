[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/TongjiFinLab/awesome-financial-time-series-forecasting)

# Awesome Papers and Codes of Large Language Models for Time Series Analysis <!-- omit in toc -->

> 💎 **Official Repository:** *Large Language Models for Time Series Analysis: A Survey*  
>
> 💎 **Update:** Actively maintained. `2025/08/25`  
>
> 💎 **Scope:** Featuring the latest and most influential papers and codes in Time Series Analysis (TSA) powered by Large Language Models (LLMs), including tasks such as forecasting, classification, anomaly detection, and imputation. Applications span finance, transportation, energy, healthcare, and more.
>
> 💎 **Collaboration:** Contributions and suggestions are highly encouraged! If you find any omissions or wish to recommend new resources, please feel free to contact us.

TSA has evolved from traditional lightweight models to Foundation Models, aiming to better capture complex temporal patterns across diverse domains. However, challenges in adaptability and generalization persist. The remarkable success of LLMs in Natural Language Processing and Computer Vision has demonstrated their powerful sequential modeling and cross-domain generalization capabilities. ***Our main contributions are as follows:***

- **Roles-Based Taxonomy & Unified Workflows**  
  We systematically categorize the roles assumed by LLMs in TSA and abstract unified workflows for each role, clarifying their core functionalities and diverse contributions to the field.

- **Mechanism-Centric Analysis of Applications**  
  We comprehensively review representative applications across multiple domains and categorize these applications based on the distinct mechanisms through which LLMs enhance domain-specific tasks, offering new perspectives and insights for the advancement of these downstream tasks.

- **Limitations & Future Directions**  
  We critically examine the key limitations and open challenges in deploying LLMs for TSA and propose prospective research directions to address these challenges and advance the field.

<div align="center">
  <img src="figs/framework.png" alt="Framework" width=800/>
</div>

<div align="center">
  <b>Figure 1: The Framework of Our Survey</b>
</div>


## 🎯 Contents <!-- omit in toc -->
- [✅ Background](#-background)
  - [Lightweight Models](#lightweight-models)
  - [Foundation Models](#foundation-models)
- [✅ Taxonomy of Roles and Unified Workflows](#-taxonomy-of-roles-and-unified-workflows)
  - [Role 1️⃣: Fine-tune-based Inference Engines](#role-1️⃣-fine-tune-based-inference-engines)
  - [Role 2️⃣: Enhancer based on TSA Methods](#role-2️⃣-enhancer-based-on-tsa-methods)
  - [Role 3️⃣: Hybrid Collaborators](#role-3️⃣-hybrid-collaborators)
- [✅ Application](#-application)
  - [Finance 💰](#finance-)
  - [Traffic 🚗](#traffic-)
  - [Energy ⚡](#energy-)
  - [Others ❤️](#others-️)
- [✅ All Thanks to Our Contributors](#-all-thanks-to-our-contributors)

## ✅ Background

### Lightweight Models

#### MLPs <!-- omit in toc -->

- **Are Transformers Effective for Time Series Forecasting?**  
  *Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu*  
  AAAI, 2023  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089) | [Code](https://github.com/cure-lab/LTSF-Linear)

- **Deep Residual Learning for Image Recognition**  
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
  CVPR, 2016.  
  [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | [Code](https://github.com/KaimingHe/deep-residual-networks)

#### CNNs <!-- omit in toc -->

- **WaveNet: A Generative Model for Raw Audio**  
  Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu  
  SSW, 2016.  
  [Paper](https://arxiv.org/pdf/1609.03499) | [Code](https://github.com/ibab/tensorflow-wavenet)

- **Scinet: Time series modeling and forecasting with sample convolution and interaction**  
  Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, Qiang Xu  
  NeurIPS, 2022.  
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/266983d0949aed78a16fa4782237dea7-Paper-Conference.pdf) | [Code](https://github.com/cure-lab/SCINet)


#### RNNs <!-- omit in toc -->

- **Segrnn: Segment recurrent neural network for long-term time series forecasting**  
  Shengsheng Lin, Weiwei Lin, Wentai Wu, Feiyu Zhao, Ruichao Mo, Haotong Zhang  
  arXiv preprint arXiv:2308.11200, 2023.  
  [Paper](https://arxiv.org/abs/2308.11200) | [Code](https://github.com/lss-1138/SegRNN)

- **Empirical evaluation of gated recurrent neural networks on sequence modeling**  
  Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio  
  NeurIPS, 2014.  
  [Paper](https://arxiv.org/pdf/1412.3555) 

- **Long short-term memory**  
  Sepp Hochreiter, Jürgen Schmidhuber  
  Neural Computation, 1997.  
  [Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)

#### Transformers <!-- omit in toc -->

- **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**  
  Haixu Zhou, Yifan Zhang, Jieqi Peng, Jianxin Wu, Ziqing Liu, Hao Li, Haoran Xu, Weijian Xu  
  AAAI, 2021.  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) | [Code](https://github.com/zhouhaoyi/Informer2020)

- **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**  
  Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng  
  arXiv preprint arXiv:2306.12151, 2023.  
  [Paper](https://openreview.net/pdf?id=JePfAI8fah) | [Code](https://github.com/thuml/iTransformer)

- **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**  
  Xiang Nie, Ailing Zeng, Muxi Chen, Qiang Xu  
  ICLR, 2023.  
  [Paper](https://openreview.net/pdf?id=Jbdc0vTOcol) | [Code](https://github.com/yuqinie98/PatchTST)

### Foundation Models

- **TimeGPT-1**  
  Garza, Azul and Mergenthaler-Canseco, Max  
  arXiv preprint arXiv:2310.03589, 2023.  
  [Paper](https://arxiv.org/abs/2310.03589) | [Code](https://github.com/Nixtla/nixtla)

- **Timer: Generative Pre-trained Transformers are Large Time Series Models**  
  Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng  
  PMLR, 2024.  
  [Paper](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24cb/liu24cb.pdf) | [Code](https://github.com/thuml/Large-Time-Series-Model)

- **A Decoder-Only Foundation Model for Time-Series Forecasting**  
  Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen  
  ICML, 2024.  
  [Paper](https://openreview.net/pdf?id=jn2iTJas6h) | [Code](https://github.com/google-research/timesfm/)

- **Lag-Llama: Towards Foundation Models for Time Series Forecasting**  
  Rasul, Kashif and Ashok, Arjun and Williams, Andrew Robert and Khorasani, Arian and Adamopoulos, George and Bhagwatkar, Rishika and Biloš, Marin and Ghonia, Hena and Hassen, Nadhir and Schneider, Anderson, et al.  
  arXiv preprint arXiv:2310.06530, 2023.  
  [Paper](https://openreview.net/pdf?id=jYluzCLFDM) | [Code](https://github.com/kashif/pytorch-transformer-ts)

- **MOMENT: A Family of Open Time-series Foundation Models**  
  Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski  
  ICML, 2024.  
  [Paper](https://arxiv.org/abs/2402.03885) | [Codes](https://github.com/moment-timeseries-foundation-model/moment)

- **Unified Training of Universal Time Series Forecasting Transformers**  
  Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, Doyen Sahoo  
  ICML, 2024.  
  [Paper](https://arxiv.org/abs/2402.02592) | [Code](https://github.com/SalesforceAIResearch/uni2ts)

## ✅ Taxonomy of Roles and Unified Workflows

<div align="center"><sub>

|**Model**|**Task**|**Role**|**Tokenization**|**Prompt**|**Semantic Alignment**|**Fine-tuning**|**Code**|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**OFA**|General|IE|`Patch-level`|`Vector-based`|`Emb-Injected`|`Endogenous(Direct)`|✅|
|**aLLM4TS**|Forecasting|IE|`Patch-level`|`Vector-based`|-|`Hybrid(Direct)`|✅|
|**PromptCast**|Forecasting|IE|`Digit-level`|`Text-based`|-|-|✅|
|**TimeLLM**|Forecasting|IE|`Patch-level`|`Vector-based`|`Emb-Injected`|`Exogenous(Direct)`|✅|
|**UniTime**|Forecasting|IE|`Patch-level`|`Vector-based`|`Emb-Injected`|`Exogenous(Direct)`|✅|
|**AutoTimes**|Forecasting|IE|`Patch-level`|`Vector-based`|-|`Hybrid(Direct)`|✅|
|**S2IP-LLM**|Forecasting|IE|`Patch-level`|`Vector-based`|`Distributional`|`Exogenous(Direct)`|✅|
|**ETP**|Classification|IE|-|`Vector-based`|`Contrastive`|`Exogenous(Direct)`|❌|
|**TENT**|Classification|IE|-|`Vector-based`|`Contrastive`|`Exogenous(Direct)`|❌|
|**Qiu et al.**|Classification|IE|-|`Vector-based`|`Distributional`|-|❌|
|**MTAM**|Classification|IE|`Patch-level`|-|`Distributional`|-|❌|
|**TimeCMA**|Forecasting|IE|`Digit-level`|`Text-based`|`Distributional`|`Exogenous(Direct)`|❌|
|**TableTime**|Classification|IE|`Digit-level`|`Text-based`|`Distributional`|`Exogenous(Direct)`|✅|
|**MedualTime**|Classification|IE|`Digit-level`|`Text-based`|`Emb-Injected`|`Exogenous(Direct)`|✅|
|**METS**|Classification|E|-|-|`Contrastive`|-|✅|
|**Xie et al.**|Forecasting|E|-|`Text-based`|-|`Multi-modal Fusion`|❌|
|**TimeReasoner**|Forecasting|E|-|`Text-based`|-|-|❌|
|**Time-R1**|Forecasting|E|-|`Text-based`|-|-|❌|
|**Time-RA**|Anomaly Etection|E|-|-|-|-|✅|
|**TEMPO**|Forecasting|IE+E|`Patch-level`|`Vector-based`|`Emb-Injected`|`Hybrid(LoRA)`|✅|
|**LLM4TS**|Forecasting|IE+E|`Patch-level`|`Vector-based`|-|`Endogenous(LoRA)`|✅|
|**TEST**|General|IE+E|`Patch-level`|`Vector-based`|`Contrastive`|`Exogenous(Direct)`|✅|
|**Chronos**|General|IE+E|`Bin-level`|`Vector-based`|-|`Exogenous(Direct)`|✅|
|**LLM-Mob**|Forecasting|IE+E|`Digit-level`|`Text-based`|-|-|❌|
|**TimeCAP**|Forecasting|IE+E|-|`Text-based`|-|`Exogenous(Direct)`|❌|
|**TS-Reasoner**|Forecasting|HC|-|`Text-based`|-|-|❌|
|**AuxMobLCast**|Forecasting|HC|`Digit-level`|`Vector-based`|-|-|❌|
|**LAMP**|Forecasting|HC|-|`Text-based`|-|-|❌|
|**LA-GCN**|Forecasting|HC|`Digit-level`|`Text-based`|-|-|❌|
|**Chen et al.**|Forecasting|HC|-|`Text-based`|-|-|❌|
|**Park et al.**|Anomaly Etection|HC|-|`Text-based`|-|-|❌|
|**Zuo et al.**|Forecasting|HC|-|`Text-based`|-|-|❌|
|**DualSG**|Forecasting|HC|-|`Text-based`|-|`Exogenous(Direct)`|✅|

</sub></div>

<sub>Note: IE stands for Inference Engine, E stands for Enhancer, IE+E stands for the combination of Inference Engine and Enhancer, and HC stands for Hybrid Collaborator.</sub>



### Role 1️⃣: Fine-tune-based Inference Engines 

#### Tokenization Approaches <!-- omit in toc -->

##### Digit-level Tokenization <!-- omit in toc -->

- **Llama 2: Open Foundation and Fine-Tuned Chat Models**  
  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample  
  arXiv, 2023.  
  [Paper](https://arxiv.org/pdf/2307.09288)

- **MedualTime: A Dual-Adapter Language Model for Medical Time Series-Text Multimodal Learning**  
  Ye, Jiexia and Zhang, Weiqi and Li, Ziyue and Li, Jia and Zhao, Meng and Tsung, Fugee  
  arXiv preprint arXiv:2406.06620, 2024.  
  [Paper](https://arxiv.org/abs/2406.06620) | [Code](https://github.com/start2020/MedualTime)


- **PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting**  
  Hao Xue, Flora D Salim  
  IEEE TKDE, 2023.  
  [Paper](https://ieeexplore.ieee.org/document/10123956) | [Code](https://github.com/haoxue2/PromptCast)


##### Patch-level Tokenization <!-- omit in toc -->

- **One fits all: Power general time series analysis by pretrained LM**  
  Tian Zhou, Peisong Niu, Liang Sun, Rong Jin, et al.  
  NeurIPS 2023.  
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/86c17de05579cde52025f9984e6e2ebb-Paper-Conference.pdf) | [Code](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

- **Multi-patch prediction: Adapting language models for time series representation learning**  
  Yuxuan Bian, Xuan Ju, Jiangtong Li, Zhijian Xu, Dawei Cheng, Qiang Xu  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Rx9GMufByc) | [Code](https://github.com/yxbian23/aLLM4TS)

- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**  
  Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Unb5CVPtae) | [Code](https://github.com/KimMeen/Time-LLM)

- **UniTime: A language-empowered unified model for cross-domain time series forecasting**  
  Xu Liu, Junfeng Hu, Yuan Li, Shizhe Diao, Yuxuan Liang, Bryan Hooi, Roger Zimmermann  
  WWW 2024.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645434) | [Code](https://github.com/liuxu77/UniTime)

##### Bin-level Tokenization <!-- omit in toc -->

- **Chronos: Learning the Language of Time Series**  
  Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Bernie Wang  
  TMLR 2024.  
  [Paper](https://openreview.net/pdf?id=gerNCVqqtR) | [Code](https://github.com/amazon-science/chronos-forecasting)

#### Prompt Engineering <!-- omit in toc -->

##### Text-based Prompt <!-- omit in toc -->

- **PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting**  
  Hao Xue, Flora D Salim  
  IEEE TKDE, 2023.  
  [Paper](https://ieeexplore.ieee.org/document/10123956) | [Code](https://github.com/haoxue2/PromptCast)

- **Leveraging language foundation models for human mobility forecasting**  
  Hao Xue, Bhanu Prakash Voutharoja, Flora D Salim  
  SIGSPATIAL 2022.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3557915.3561026)

- **Where would I go next? Large language models as human mobility predictors**  
  Xinglei Wang, Meng Fang, Zichao Zeng, Tao Cheng  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2308.15197) | [Code](https://github.com/xlwang233/LLM-Mob)

- **LST-Prompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting**  
  Haoxin Liu, Zhiyuan Zhao, Jindong Wang, Harshavardhan Kamarthi, B Aditya Prakash  
  ACL 2024.  
  [Paper](https://aclanthology.org/2024.findings-acl.466.pdf) | [Code](https://github.com/AdityaLab/lstprompt)

- **TableTime: Reformulating Time Series Classification as Training-Free Table Understanding with Large Language Models**  
  Jiahao Wang, Mingyue Cheng, Qingyang Mao, Yitong Zhou, Feiyang Xu, Xin Li  
  arXiv 2024.  
  [Paper](https://arxiv.org/abs/2411.15737)

- **The Wall Street Neophyte: A Zero-Shot Analysis of ChatGPT Over MultiModal Stock Movement Prediction Challenges**  
  Qianqian Xie, Weiguang Han, Yanzhao Lai, Min Peng, Jimin Huang  
  arXiv 2023.  
  [Paper](https://arxiv.org/pdf/2304.05351)

##### Vector-based (Soft) Prompt <!-- omit in toc -->

- **One fits all: Power general time series analysis by pretrained LM**  
  Tian Zhou, Peisong Niu, Liang Sun, Rong Jin, et al.  
  NeurIPS 2023.  
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/86c17de05579cde52025f9984e6e2ebb-Paper-Conference.pdf) | [Code](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

- **TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series**  
  Chenxi Sun, Hongyan Li, Yaliang Li, Shenda Hong  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Tuh4nZVb0g)

- **TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting**  
  Defu Cao, Furong Jia, Sercan Ö. Arik, Tomas Pfister, Yixiang Zheng, Wen Ye, Yan Liu  
  ICLR 2024.  
  [Paper](https://arxiv.org/abs/2310.04948) | [Code](https://github.com/DC-research/TEMPO)

- **S²IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting**  
  Zijie Pan, Yushan Jiang, Sahil Garg, Anderson Schneider, Yuriy Nevmyvaka, Dongjin Song  
  ICML 2024.  
  [Paper](https://openreview.net/pdf?id=qwQVV5R8Y7) | [Code](https://github.com/panzijie825/S2IP-LLM)

- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**  
  Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Unb5CVPtae) | [Code](https://github.com/KimMeen/Time-LLM)

- **AutoTimes: Autoregressive time series forecasters via large language models**  
  Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, Mingsheng Long  
  NeurIPS 2024.  
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/dcf88cbc8d01ce7309b83d0ebaeb9d29-Paper-Conference.pdf) | [Code](https://github.com/thuml/AutoTimes)

- **Domain-Oriented Time Series Inference Agents for Reasoning and Automated Analysis**  
  Wen Ye, Wei Yang, Defu Cao, Yizhou Zhang, Lumingyuan Tang, Jie Cai, Yan Liu  
  arXiv 2024.  
  [Paper](https://arxiv.org/abs/2410.04047)

#### Semantic Alignment <!-- omit in toc -->

##### Contrastive Alignment <!-- omit in toc -->

- **ETP: Learning transferable ECG representations via ECG-text pre-training**  
  Che Liu, Zhongwei Wan, Sibo Cheng, Mi Zhang, Rossella Arcucci  
  ICASSP 2024.  
  [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10446742)

- **TimeCMA: Towards LLM-empowered multivariate time series forecasting via cross-modality alignment**  
  Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, Rui Zhao  
  AAAI 2025.  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34067/36222) | [Code](https://github.com/ChenxiLiu-HNU/TimeCMA)

- **Can brain signals reveal inner alignment with human languages?**  
  Jielin Qiu, William Han, Jiacheng Zhu, Mengdi Xu, Douglas Weber, Bo Li, Ding Zhao  
  EMNLP 2023.  
  [Paper](https://aclanthology.org/2023.findings-emnlp.120.pdf) | [Code](https://github.com/Jielin-Qiu/EEG_Language_Alignment)

##### Distributional Alignment <!-- omit in toc -->

- **Transfer knowledge from natural language to electrocardiography: Can we detect cardiovascular disease through language models?**  
  Jielin Qiu, William Han, Jiacheng Zhu, Mengdi Xu, Michael Rosenberg, Emerson Liu, Douglas Weber, Ding Zhao  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2301.09017) | [Code](https://github.com/Jielin-Qiu/Transfer_Knowledge_from_Language_to_ECG)

- **S²IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting**  
  Zijie Pan, Yushan Jiang, Sahil Garg, Anderson Schneider, Yuriy Nevmyvaka, Dongjin Song  
  ICML 2024.  
  [Paper](https://openreview.net/pdf?id=qwQVV5R8Y7) | [Code](https://github.com/panzijie825/S2IP-LLM)

##### Embedding-Injected Alignment <!-- omit in toc -->

- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**  
  Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Unb5CVPtae) | [Code](https://github.com/KimMeen/Time-LLM)

- **TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series**  
  Chenxi Sun, Hongyan Li, Yaliang Li, Shenda Hong  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Tuh4nZVb0g)

- **Tent: Connect language models with IoT sensors for zero-shot activity recognition**  
  Yunjiao Zhou, Jianfei Yang, Han Zou, Lihua Xie  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2311.08245)

#### Fine-tuning <!-- omit in toc -->

##### Parameters Selection <!-- omit in toc -->

###### Endogenous Parameters <!-- omit in toc -->

- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**  
  Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Unb5CVPtae) | [Code](https://github.com/KimMeen/Time-LLM)

- **Multi-patch prediction: Adapting language models for time series representation learning**  
  Yuxuan Bian, Xuan Ju, Jiangtong Li, Zhijian Xu, Dawei Cheng, Qiang Xu  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Rx9GMufByc) | [Code](https://github.com/yxbian23/aLLM4TS)

###### Exogenous Parameters <!-- omit in toc -->

- **S²IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting**  
  Zijie Pan, Yushan Jiang, Sahil Garg, Anderson Schneider, Yuriy Nevmyvaka, Dongjin Song  
  ICML 2024.  
  [Paper](https://openreview.net/pdf?id=qwQVV5R8Y7) | [Code](https://github.com/panzijie825/S2IP-LLM)

- **TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series**  
  Chenxi Sun, Hongyan Li, Yaliang Li, Shenda Hong  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Tuh4nZVb0g)

###### Hybrid Parameters <!-- omit in toc -->

- **TimeCMA: Towards LLM-empowered multivariate time series forecasting via cross-modality alignment**  
  Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, Rui Zhao  
  AAAI 2025.  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34067/36222) | [Code](https://github.com/ChenxiLiu-HNU/TimeCMA)

- **UniTime: A language-empowered unified model for cross-domain time series forecasting**  
  Xu Liu, Junfeng Hu, Yuan Li, Shizhe Diao, Yuxuan Liang, Bryan Hooi, Roger Zimmermann  
  WWW 2024.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645434) | [Code](https://github.com/liuxu77/UniTime)

##### Fine-tuning Strategies Selection <!-- omit in toc -->

###### Direct Fine-tuning <!-- omit in toc -->

- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**  
  Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Unb5CVPtae) | [Code](https://github.com/KimMeen/Time-LLM)

- **A Decoder-Only Foundation Model for Time-Series Forecasting**  
  Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen  
  ICML, 2024.  
  [Paper](https://openreview.net/pdf?id=jn2iTJas6h) | [Code](https://github.com/google-research/timesfm/)

- **Multi-patch prediction: Adapting language models for time series representation learning**  
  Yuxuan Bian, Xuan Ju, Jiangtong Li, Zhijian Xu, Dawei Cheng, Qiang Xu  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Rx9GMufByc) | [Code](https://github.com/yxbian23/aLLM4TS)

###### LoRA-based Fine-tuning <!-- omit in toc -->

- **UniTime: A language-empowered unified model for cross-domain time series forecasting**  
  Xu Liu, Junfeng Hu, Yuan Li, Shizhe Diao, Yuxuan Liang, Bryan Hooi, Roger Zimmermann  
  WWW 2024.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645434) | [Code](https://github.com/liuxu77/UniTime)

- **TimeCMA: Towards LLM-empowered multivariate time series forecasting via cross-modality alignment**  
  Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, Rui Zhao  
  AAAI 2025.  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34067/36222) | [Code](https://github.com/ChenxiLiu-HNU/TimeCMA)

- **S²IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting**  
  Zijie Pan, Yushan Jiang, Sahil Garg, Anderson Schneider, Yuriy Nevmyvaka, Dongjin Song  
  ICML 2024.  
  [Paper](https://openreview.net/pdf?id=qwQVV5R8Y7) | [Code](https://github.com/panzijie825/S2IP-LLM)

### Role 2️⃣: Enhancer based on TSA Methods

#### Enhancement of Time Series Data <!-- omit in toc -->

##### Self-Supervised Learning <!-- omit in toc -->

- **LLM4TS: Aligning pre-trained LLMs as data-efficient time-series forecasters**  
  Ching Chang, Wei-Yao Wang, Wen-Chih Peng, Tien-Fu Chen  
  ACM Transactions on Intelligent Systems and Technology 2025.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3719207) | [Code](https://github.com/blacksnail789521/LLM4TS)

- **Multi-patch prediction: Adapting language models for time series representation learning**  
  Yuxuan Bian, Xuan Ju, Jiangtong Li, Zhijian Xu, Dawei Cheng, Qiang Xu  
  ICLR 2024.  
  [Paper](https://openreview.net/pdf?id=Rx9GMufByc) | [Code](https://github.com/yxbian23/aLLM4TS)

##### Synthetic Data Generation <!-- omit in toc -->

- **Chronos: Learning the Language of Time Series**  
  Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Bernie Wang  
  TMLR 2024.  
  [Paper](https://openreview.net/pdf?id=gerNCVqqtR) | [Code](https://github.com/amazon-science/chronos-forecasting)

- **TimeCAP: Learning to contextualize, augment, and predict time series events with large language model agents**  
  Geon Lee, Wenchao Yu, Kijung Shin, Wei Cheng, Haifeng Chen  
  AAAI 2025.  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33989/36144) | [Code](https://github.com/geon0325/TimeCAP)

##### Multi-Modal Data Fusion <!-- omit in toc -->

- **Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models**  
  Tan Xie, Zhi Da  
  arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07619)]

- **Frozen language model helps ECG zero-shot learning**  
  Jun Li, Che Liu, Sibo Cheng, Rossella Arcucci, Shenda Hong  
  arXiv 2024.  
  [Paper](https://proceedings.mlr.press/v227/li24a/li24a.pdf)

- **TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting**  
  Defu Cao, Furong Jia, Sercan Ö. Arik, Tomas Pfister, Yixiang Zheng, Wen Ye, Yan Liu  
  ICLR 2024.  
  [Paper](https://arxiv.org/abs/2310.04948) | [Code](https://github.com/DC-research/TEMPO)

- **GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting**  
  Furong Jia, Kevin Wang, Yixiang Zheng, Defu Cao, Yan Liu  
  AAAI, 2024.  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/30383)

#### Enhancement of Interpretability <!-- omit in toc -->

- **Can "Slow-thinking" LLMs Make Time Series Predictions More Reliable? Enhancing LLM-based Time Series Forecasting via Chain-of-Thought Prompting**  
  Shuai Wang, Qing Li, Chenyang Shang, Yushu Chen, Zhenyu Liu, Xiang Li, Shenda Hong  
  arXiv 2025.  
  [Paper](https://arxiv.org/pdf/2505.24511) | [Code](https://github.com/realwangjiahao/TimeReasoner)

- **Time-R1: Towards Comprehensive Temporal Reasoning in LLMs**  
  Zijia Liu, Peixuan Han, Haofei Yu, Haoru Li, Jiaxuan You  
  arXiv 2025.  
  [Paper](https://arxiv.org/pdf/2505.13508)

- **Where would I go next? Large language models as human mobility predictors**  
  Xinglei Wang, Meng Fang, Zichao Zeng, Tao Cheng  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2308.15197) | [Code](https://github.com/xlwang233/LLM-Mob)

- **Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback**  
  Yiyuan Yang, Zichuan Liu, Lei Song, Kai Ying, Zhiguang Wang, Tom Bamford, Svitlana Vyetrenko, Jiang Bian, Qingsong Wen  
  arXiv 2025.  
  [Paper](https://arxiv.org/pdf/2507.15066?)

### Role 3️⃣: Hybrid Collaborators

- **Domain-Oriented Time Series Inference Agents for Reasoning and Automated Analysis**  
  Wen Ye, Wei Yang, Defu Cao, Yizhou Zhang, Lumingyuan Tang, Jie Cai, Yan Liu  
  arXiv 2024.  
  [Paper](https://arxiv.org/abs/2410.04047)

- **Language models can improve event prediction by few-shot abductive reasoning**  
  Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei  
  NeurIPS 2023.  
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/5e5fd18f863cbe6d8ae392a93fd271c9-Paper-Conference.pdf)

- **Leveraging language foundation models for human mobility forecasting**  
  Hao Xue, Bhanu Prakash Voutharoja, Flora D Salim  
  SIGSPATIAL 2022.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3557915.3561026)

- **ChatGPT Informed Graph Neural Network for Stock Movement Prediction**  
  Zihan Chen, Lei Nico Zheng, Cheng Lu, Jialu Yuan, Di Zhu  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2306.03763) | [Code](https://github.com/ZihanChen1995/ChatGPT-GNN-StockPredict)

- **Language knowledge-assisted representation learning for skeleton-based action recognition**  
  Haojun Xu, Yan Gao, Zheng Hui, Jie Li, Xinbo Gao  
  IEEE Transactions on Multimedia 2025.  
  [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10891636)

- **DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework**  
  Kuiye Ding, Fanda Fan, Yao Wang, Xiaorui Wang, Luqi Gong, Yishan Jiang, others  
  arXiv 2025.  
  [Paper](https://arxiv.org/pdf/2507.21830) | [Code](https://github.com/BenchCouncil/DualSG)

- **Enhancing Anomaly Detection in Financial Markets with an LLM-based Multi-Agent Framework**  
  Taejin Park  
  arXiv 2024.  
  [Paper](https://arxiv.org/abs/2403.19735)

- **Large Language Model-Empowered Interactive Load Forecasting**  
  Yu Zuo, Dalin Qin, Yi Wang  
  arXiv 2025.  
  [Paper](https://arxiv.org/pdf/2505.16577?)

## ✅ Application

<div align="center">
  <img src="figs/application_figs.png" alt="Application" width=800/>
</div>

<div align="center">
  <b>Figure 2: An overview of the application of LLMs for TSA.</b>
</div>

### Finance 💰

#### Stock Movement (Trend) Forecasting <!-- omit in toc -->

- **ChatGPT Informed Graph Neural Network for Stock Movement Prediction**  
  Zihan Chen, Lei Nico Zheng, Cheng Lu, Jialu Yuan, Di Zhu  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2306.03763) | [Code](https://github.com/ZihanChen1995/ChatGPT-GNN-StockPredict)

- **Integrating Stock Features and Global Information via Large Language Models for Enhanced Stock Return Prediction**  
  Enhao Zhang, Lingxuan Zhao, Xinran Li, Zhengyang Li, Yue Zhang, Yan Zhou, Yongfeng Zhang  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2310.05627)

- **Ploutos: Towards Interpretable Stock Movement Prediction with Financial Large Language Model**  
  Hang Tong, Xinyi Du, Jianguo Li  
  arXiv 2024.  
  [Paper](https://arxiv.org/abs/2403.00782)

- **Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models**  
  Duxin Xie, Jingru Zhang, Hui Wang, Yongqiang Chu, Jiayu Li  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2304.07619)

- **LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction**  
  Meiyun Wang, Kiyoshi Izumi, Hiroki Sakaji  
  ACL 2024.  
  [Paper](https://aclanthology.org/2024.findings-acl.185.pdf)

- **Learning to generate explainable stock predictions using self-reflective large language models**  
  Kelvin JL Koa, Yunshan Ma, Ritchie Ng, Tat-Seng Chua  
  ACM Web Conference 2024.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645611) | [Code](https://github.com/koa-fin/sep)

#### Stock Price Forecasting  <!-- omit in toc -->

- **Temporal Data Meets LLM -- Explainable Financial Time Series Forecasting**  
  Xinli Yu, Zheng Chen, Yuan Gao, Zongyu Dai, Qianqian Xie, Jimin Huang  
  EMNLP 2023.  
  [Paper](https://aclanthology.org/2023.emnlp-main.193.pdf) | [Code](https://github.com/bnewm0609/qa-decontext/tree/emnlp)

- **Leveraging Vision-Language Models for Granular Market Change Prediction**  
  Christopher Wimmer, Navid Rekabsaz  
  arXiv 2023.  
  [Paper](https://arxiv.org/pdf/2301.10166)

- **StockTime: A Time Series Specialized Large Language Model Architecture for Stock Price Prediction**  
  Shengkun Wang, Taoran Ji, Linhan Wang, Yanshen Sun, Shang-Ching Liu, Amit Kumar, Chang-Tien Lu  
  arXiv 2024.  
  [Paper](https://arxiv.org/pdf/2409.08281?)

- **Fine-Tuning Large Language Models for Stock Return Prediction Using Newsflow**  
  Tian Guo, Emmanuel Hauptmann  
  arXiv 2024.  
  [Paper](https://arxiv.org/pdf/2407.18103?)

### Traffic 🚗

#### Traffic Flow Forecasting <!-- omit in toc -->

- **How can large language models understand spatial-temporal data?**  
  Lei Liu, Shuo Yu, Runze Wang, Zhenxun Ma, Yanming Shen  
  arXiv 2024.  
  [Paper](https://arxiv.org/pdf/2401.14192)

- **LLM-TFP: Integrating large language models with spatio-temporal features for urban traffic flow prediction**  
  Haitao Cheng, Zibin Gong, Chang Wang  
  Applied Soft Computing, 2025.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S1568494625004855)

- **Edge computing enabled large-scale traffic flow prediction with GPT in intelligent autonomous transport system for 6G network**  
  Yi Rong, Yingchi Mao, Huajun Cui, Xiaoming He, Mingkai Chen  
  IEEE Transactions on Intelligent Transportation Systems (TITS), 2024.  
  [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10682107)

- **Spatial-Temporal Large Language Model for Traffic Prediction**  
  Chenxi Liu, Sun Yang, Qianxiong Xu, Zhishuai Li, Cheng Long, Ziyue Li, Rui Zhao  
  arXiv 2024.  
  [Paper](https://arxiv.org/pdf/2401.10134) | [Code](https://github.com/ChenxiLiu-HNU/ST-LLM)

- **ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction**  
  Chenxi Liu, Kethmi Hirushini Hettige, Qianxiong Xu, Cheng Long, Shili Xiang, Gao Cong, Ziyue Li, Rui Zhao  
  IEEE Transactions on Knowledge and Data Engineering, 2025.  
  [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11005661)

- **GPT4TFP: Spatio-temporal fusion large language model for traffic flow prediction**  
  Yiwu Xu, Mengchi Liu  
  Neurocomputing, 2025.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0925231225002346)

- **TPLLM: A Traffic Prediction Framework Based on Pretrained Large Language Models**  
  Yilong Ren, Yue Chen, Shuai Liu, Boyue Wang, Haiyang Yu, Zhiyuan Liu  
  arXiv 2024.  
  [Paper](https://arxiv.org/pdf/2403.02221)

- **TrafficBERT: Pre-trained model with large-scale permuted traffic data for long-term traffic forecasting**  
  Daejin Kim, Youngin Cho, Dongmin Kim, Cheonbok Park, Jaegul Choo  
  Expert Systems with Applications, 2021.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0957417421011179)

- **Embracing large language models in traffic flow forecasting**  
  Yusheng Zhao, Xiao Luo, Haomin Wen, Zhiping Xiao, Wei Ju, Ming Zhang  
  arXiv 2024.  
  [Paper](https://arxiv.org/pdf/2412.12201)

- **TrafficGPT: Viewing, processing and interacting with traffic foundation models**  
  Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai, Baozhen Yao  
  Transport Policy, 2024.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0967070X24000726) | [Code](https://github.com/lijlansg/TrafficGPT)

#### Human Mobility Forecasting <!-- omit in toc -->

- **Exploring large language models for human mobility prediction under public events**  
  Yuebing Liang, Yichao Liu, Xiaohan Wang, Zhan Zhao  
  Computers, Environment and Urban Systems, 2024.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0198971524000826)

- **Mobility-llm: Learning visiting intentions and travel preference from human mobility data with large language models**  
  Letian Gong, Yan Lin, Yiwen Lu, Xuedi Han, Yichen Liu, Shengnan Guo, Youfang Lin, Huaiyu Wan, et al.  
  Advances in NeurIPS, 2024.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0198971524000826)

- **Where would I go next? Large language models as human mobility predictors**  
  Xinglei Wang, Meng Fang, Zichao Zeng, Tao Cheng  
  arXiv 2023.  
  [Paper](https://arxiv.org/abs/2308.15197) | [Code](https://github.com/xlwang233/LLM-Mob)

- **Toward interactive next location prediction driven by large language models**  
  Yong Chen, Ben Chi, Chuanjia Li, Yuliang Zhang, Chenlei Liao, Xiqun Chen, Na Xie  
  IEEE Transactions on Computational Social Systems, 2025.  
  [Paper](https://ieeexplore.ieee.org/abstract/document/10835157)

- **Large Language Models for Spatial Trajectory Patterns Mining**  
  Zheng Zhang, Hossein Amiri, Zhenke Liu, Liang Zhao, Andreas Zuefle  
  SIGSPATIAL, 2024.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3681765.3698467) | [Code](https://github.com/onspatial/LLM-outlier-detection)

- **Leveraging language foundation models for human mobility forecasting**  
  Hao Xue, Bhanu Prakash Voutharoja, Flora D Salim  
  SIGSPATIAL, 2022.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3557915.3561026)

- **UrbanLLM: Autonomous Urban Activity Planning and Management with Large Language Models**  
  Yifei Jiang, Xinyan Zhu, Jiayu Fan, Hua Wei  
  arXiv, 2024.  
  [Paper](https://arxiv.org/pdf/2406.12360) | [Code](https://github.com/JIANGYUE61610306/UrbanLLM/tree/main)

### Energy ⚡

#### Power Load Forecasting <!-- omit in toc -->

- **MMGPT4LF: Leveraging an optimized pre-trained GPT-2 model with multi-modal cross-attention for load forecasting**  
  Mingyang Gao, Suyang Zhou, Wei Gu, Zhi Wu, Haiquan Liu, Aihua Zhou, Xinliang Wang  
  Applied Energy, 2025.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0306261925006956)

- **A general framework for load forecasting based on pre-trained large language model**  
  Mingyang Gao, Suyang Zhou, Wei Gu, Zhi Wu, Haiquan Liu, Aihua Zhou  
  arXiv 2024.  
  [Paper](https://arxiv.org/pdf/2406.11336)

- **Utilizing language models for energy load forecasting**  
  Hao Xue, Flora D. Salim  
  In Proceedings of the 10th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation, 2023.  
  [Paper](https://dl.acm.org/doi/pdf/10.1145/3600100.3623730)

- **Empower pre-trained large language models for building-level load forecasting**  
  Yating Zhou, Meng Wang  
  IEEE Transactions on Power Systems, 2025.  
  [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10917006)

- **TimeGPT in load forecasting: A large time series model perspective**  
  Wenlong Liao, Shouxiang Wang, Dechang Yang, Zhe Yang, Jiannong Fang, Christian Rehtanz, Fernando Porté-Agel  
  Applied Energy, 2025.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0306261925006956)

- **Large Language Model-Empowered Interactive Load Forecasting**  
  Yu Zuo, Dalin Qin, Yi Wang  
  arXiv, 2025.  
  [Paper](https://arxiv.org/pdf/2505.16577)

#### Climate (Weather) Forecasting <!-- omit in toc -->

- **WeatherQA: Can multimodal language models reason about severe weather?**  
  Chengqian Ma, Zhanxiang Hua, Alexandra Anderson-Frey, Vikram Iyer, Xin Liu, Lianhui Qin  
  arXiv preprint arXiv:2406.11217, 2024.  
  [Paper](https://arxiv.org/pdf/2406.11217) | [Code](https://github.com/chengqianma/WeatherQA)

- **ClimaX: A foundation model for weather and climate**  
  Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, Aditya Grover  
  In CIML, Volume 202, Pages 25904--25938, July 2023.  
  [Paper](https://proceedings.mlr.press/v202/nguyen23a/nguyen23a.pdf) | [Code](https://github.com/microsoft/ClimaX)

- **Climatellm: Efficient weather forecasting via frequency-aware large language models**  
  Shixuan Li, Wei Yang, Peiyu Zhang, Xiongye Xiao, Defu Cao, Yuehan Qin, Xiaole Zhang, Yue Zhao, Paul Bogdan  
  arXiv preprint arXiv:2502.11059, 2025.  
  [Paper](https://arxiv.org/pdf/2502.11059)

- **STELLM: Spatio-temporal enhanced pre-trained large language model for wind speed forecasting**  
  Tangjie Wu, Qiang Ling  
  Applied Energy, Volume 375, Pages 124034, 2024.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S030626192401417X)

- **GLALLM: Adapting LLMs for spatio-temporal wind speed forecasting via global-local aware modeling**  
  Tangjie Wu, Qiang Ling  
  Knowledge-Based Systems, Pages 113739, 2025.  
  [Paper](https://www.sciencedirect.com/science/article/pii/S0950705125007853)

- **EF-LLM: Energy forecasting LLM with AI-assisted automation, enhanced sparse prediction, hallucination detection**  
  Zihang Qiu, Chaojie Li, Zhongyang Wang, Renyou Xie, Borui Zhang, Huadong Mo, Guo Chen, Zhaoyang Dong  
  arXiv preprint arXiv:2411.00852, 2024.  
  [Paper](https://arxiv.org/pdf/2411.00852)

### Others ❤️ 

- **Frozen language model helps ECG zero-shot learning**  
  Jun Li, Che Liu, Sibo Cheng, Rossella Arcucci, Shenda Hong  
  arXiv, 2024.  
  [Paper](https://proceedings.mlr.press/v227/li24a/li24a.pdf)

- **Health system-scale language models are all-purpose prediction engines**  
  Jiang, Lavender Yao; Liu, Xujin Chris; Nejatian, Nima Pour; Nasir-Moin, Mustafa; Wang, Duo; Abidin, Anas; Eaton, Kevin; Riina, Howard Antony; Laufer, Ilya; Punjabi, Paawan; 等  
  Nature, 2023.  
  [Paper](https://www.nature.com/articles/s41586-023-06160-y.pdf)

- **MedualTime: A Dual-Adapter Language Model for Medical Time Series-Text Multimodal Learning**  
  Ye, Jiexia; Zhang, Weiqi; Li, Ziyue; Li, Jia; Zhao, Meng; Tsung, Fugee  
  arXiv preprint arXiv:2406.06620, 2024.  
  [Paper](https://arxiv.org/abs/2406.06620) | [Code](https://github.com/start2020/MedualTime)

## ✅ All Thanks to Our Contributors

<a href="https://github.com/TongjiFinLab/awesome-financial-time-series-forecasting/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/awesome-financial-time-series-forecasting" />
</a>
