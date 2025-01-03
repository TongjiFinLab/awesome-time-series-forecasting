[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/TongjiFinLab/awesome-financial-time-series-forecasting) 

# Awesome Financial Time Series Forecasting Papers and Codes


+ **Update**: This repository is actively updated.  `2024/9/24`
+ **Collection**: We've compiled a comprehensive list of awesome financial time series forecasting papers and codes.
+ **Collaborate**: If there’s anything missing or if you'd like to contribute, please don't hesitate to get in touch!

## Contents

- [Awesome Financial Time Series Forecasting Papers and Codes](#awesome-financial-time-series-forecasting-papers-and-codes)
  - [Contents](#contents)
  - [LLM-based Financial Time Series Forecasting Models](#llm-based-financial-time-series-forecasting-models)
  - [LLM-based Financial Models](#llm-based-financial-models)
    - [AI Agents for Finance](#ai-agents-for-finance)
    - [Stock Prediction and Analysis](#stock-prediction-and-analysis)
    - [Benchmarks and Datasets](#benchmarks-and-datasets)
    - [General Financial Large Language Models](#general-financial-large-language-models)
  - [Graph Neural Network-based Models](#graph-neural-network-based-models)
  - [Reinforcement Learning-based Models](#reinforcement-learning-based-models)
  - [Transformer-based Models](#transformer-based-models)
  - [Generative Methods based Models](#generative-methods-based-models)
  - [Classical Time Series Models](#classical-time-series-models)
  - [Quantitative Open Sourced Framework](#quantitative-open-sourced-framework)
  - [Survey](#survey)
  - [All Thanks to Our Contributors :](#all-thanks-to-our-contributors-)


## LLM-based Financial Time Series Forecasting Models

**TEMPO: PROMPT-BASED GENERATIVE PRE-TRAINED TRANSFORMER FOR TIME SERIES FORECASTING** 

*Defu Cao, Furong Jia, Sercan O. Arık, Tomas Pfister, Yixiang Zheng, Wen Ye, Yan Liu* 

ICLR 2024. [[Paper](https://openreview.net/forum?id=YH5w12OUuU)] | [[Codes](https://github.com/DC-research/TEMPO)]

**GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting** 

*Furong Jia, Kevin Wang, Yixiang Zheng, Defu Cao, Yan Liu* 

AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/30383)]

**Multi-Patch Prediction: Adapting Language Models for Time Series Representation Learning** 

*Yuxuan Bian, Xuan Ju, Jiangtong Li, Zhijian Xu, Dawei Cheng, Qiang Xu* 

ICML 2024. [[Paper](https://openreview.net/forum?id=Rx9GMufByc)] | [[Codes](https://github.com/yxbian23/aLLM4TS)]

**MOMENT: A Family of Open Time-series Foundation Models**

*Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski*

ICML 2024. [[Paper](https://arxiv.org/abs/2402.03885)] | [[Codes](https://github.com/moment-timeseries-foundation-model/moment)]

**TIME-LLM: TIME SERIES FORECASTING BY REPROGRAMMING LARGE LANGUAGE MODELS**

*Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, et.al.*

ICLR 2024. [[Paper](https://arxiv.org/abs/2310.01728)] | [[Codes](https://github.com/KimMeen/Time-LLM)]

**Timer: Generative Pre-trained Transformers Are Large Time Series Models.**

*Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, Mingsheng Long*

ICML 2024. [[Paper](https://arxiv.org/abs/2402.02368)] | [[Codes](https://github.com/thuml/Large-Time-Series-Model)]

**Unified Training of Universal Time Series Forecasting Transformers**

*Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, Doyen Sahoo*

ICML 2024. [[Paper](https://arxiv.org/abs/2402.02592)] | [[Codes](https://github.com/redoules/moirai)]

**Are Language Models Actually Useful for Time Series Forecasting?**

*Mingtian Tan, Mike A. Merrill, Vinayak Gupta, Tim Althoff, Thomas Hartvigsen*

Arxiv 2024. [[Paper](https://arxiv.org/abs/2406.16964)]

**TIME-LLM: TIME SERIES FORECASTING BY REPROGRAMMING LARGE LANGUAGE MODELS**

*Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, et.al*

ICLR 2024. [[Paper](https://arxiv.org/abs/2310.01728v1)] | [[Codes](https://github.com/KimMeen/Time-LLM)]


## LLM-based Financial Models

### AI Agents for Finance

  **FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making**
*Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yupeng Cao, Qianqian Xie, et.al*
**NeurIPS 2024**. [[Paper](https://arxiv.org/abs/2407.06567)] | [[Codes](https://github.com/The-FinAI/FinCon)]


**FinRobot: AI Agent for Equity Research and Valuation with Large Language Models**
*Tianyu Zhou, Pinqiao Wang, Yilin Wu, Hongyang Yang*
ICAIF 2024. [[Paper](https://arxiv.org/abs/2411.08804)] | [[Codes](https://github.com/AI4Finance-Foundation/FinRobot)]

**A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist**
*Wentao Zhang, Lingxuan Zhao, Haochong Xia, Shuo Sun, Jiaze Sun, Molei Qin, Bo An, et.al*
KDD 2024. [[Paper](https://arxiv.org/abs/2402.18485)]

### Stock Prediction and Analysis
**LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction**
*Meiyun Wang, Kiyoshi Izumi, Hiroki Sakaji*
ACL 2024. [[Paper](https://aclanthology.org/2024.findings-acl.185.pdf)]

**Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models**
*Kelvin J.L. Koa, Yunshan Ma,Ritchie Ng, Tat-Seng Chua*
WWW 2024. [[Paper](https://arxiv.org/abs/2402.03659)] | [[Codes](https://github.com/koa-fin/sep)]

**S2IP-LLM: Semantic Space Informed Prompt Learning with LLM forTime Series Forecasting** 
*Zijie Pan, Yushan Jiang, Sahil Garg, Anderson Schneider, Yuriy Nevmyvaka, Dongjin Song* 
ICML 2024. [[Paper](https://arxiv.org/abs/2403.05798)] | [[Codes](https://github.com/panzijie825/S2IP-LLM)]

### Benchmarks and Datasets

**CFGPT: Chinese Financial Assistant with Large Language Model**
*Jiangtong Li, Yuxuan Bian, Guoxuan Wang, Yang Lei, Dawei Cheng, Zhijun Ding, Changjun Jiang*
$2024$. [[Paper](https://arxiv.org/pdf/2309.10654)] | [[Codes](https://github.com/TongjiFinLab/CFBenchmark)]

**RA-CFGPT: Chinese financial assistant with retrieval-augmented large language model**
*Jiangtong Li, Yang Lei, Yuxuan Bian, Dawei Cheng, Zhijun Ding, Changjun Jiang*
FCS 2024. [[Paper](https://link.springer.com/article/10.1007/s11704-024-31018-5)]

**CSPRD: A Financial Policy Retrieval Dataset for Chinese Stock Market**
*Jinyuan Wang, Zhong Wang, Zeyang Zhu, Jinhao Xie, Yong Yu, Yongjian Fei, Yue Huang, Dawei Cheng, Hai Zhao*
DEXA 2024. [[Paper](https://arxiv.org/abs/2309.04389)] | [[Codes](https://github.com/noewangjy/csprd_dataset)]

**FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets**
*Neng Wang, Hongyang Yang, Christina Dan Wang*
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.04793)] | [[Codes](https://github.com/AI4Finance-Foundation/FinGPT)]

**FinGPT: Democratizing Internet-scale Data for Financial Large Language Models**
*Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, Daochen Zha*
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2307.10485)] | [[Codes](https://github.com/AI4Finance-Foundation/FinGPT)]


### General Financial Large Language Models

**BloombergGPT: A Large Language Model for Finance**
*Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Gideon Mann, et.al*
$2023$. [[Paper](https://arxiv.org/abs/2303.17564)]


## Graph Neural Network-based Models

**Automatic De-Biased Temporal-Relational Modeling for Stock Investment Recommendation**

*Weijun Chen, Shun Li, Xipu Yu, Heyuan Wang, Wei Chen, Tengjiao Wang*

IJCAI 2024. [[Paper](https://www.ijcai.org/proceedings/2024/221)]

**MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction**

*Hao Qian, Hongting Zhou, Qian Zhao, Hao Chen, Hongxiang Yao, Jingwei Wang, Ziqi Liu, Fei Yu, Zhiqiang Zhang, Jun Zhou*

AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29381)]

**ECHO-GL: Earnings Calls-Driven Heterogeneous Graph Learning for Stock Movement Prediction**

*Mengpu Liu, Mengying Zhu, Xiuyuan Wang, Guofang Ma, Jianwei Yin, Xiaolin Zheng*

AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29305)] | [[Codes](https://github.com/pupu0302/ECHOGL)]

**TCGPN: Temporal-Correlation Graph Pre-trained Network for Stock Forecasting**

*Wenbo Yan, Ying Tan*

[[Paper](https://arxiv.org/abs/2407.18519)]

**Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction**

*Sheng Xiang, Dawei Cheng, Chencheng Shang, Ying Zhang, Yuqi Liang*

CIKM 2022. [[Paper](https://arxiv.org/abs/2305.08740)] | [[Codes](https://github.com/finint/THGNN)]

**Relational Temporal Graph Convolutional Networks for Ranking-Based Stock Prediction**

*Zetao Zheng, Jie Shao, Jia Zhu, Heng Tao Shen*

ICDE 2023. [[Paper](https://ieeexplore.ieee.org/document/10184655)] | [[Codes](https://github.com/zhengzetao/RTGCN)]

**Temporal-Relational hypergraph tri-Attention networks for stock trend prediction**

*Chaoran Cui, Xiaojie Li, Chunyun Zhang, Weili Guan, Meng Wang*

Pattern Recognition 2023. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323004570)] | [[Codes](https://github.com/lixiaojieff/HGTAN)]

**Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction**

*Sheng Xiang, Dawei Cheng, Chencheng Shang, Ying Zhang, Yuqi Liang*

CIKM 2022. [[Paper](https://arxiv.org/abs/2305.08740)] | [[Codes](https://github.com/TongjiFinLab/THGNN)]

**Financial time series forecasting with multi-modality graph neural network**

*Dawei Cheng, Fangzhou Yang, Sheng Xiang, Jin Liu*

Pattern Recognition 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S003132032100399X)] | [[Codes](https://github.com/finint/MAGNN)]

**Hierarchical Adaptive Temporal-Relational Modeling for Stock Trend Prediction**

*Heyuan Wang, Shun Li, Tengjiao Wang, Jiayi Zheng*

IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/508)] | [[Codes](https://github.com/lixiaojieff/HGTAN)]

**REST: Relational Event-driven Stock Trend Forecasting**

*Wentao Xu, Weiqing Liu, Chang Xu, Jiang Bian, Jian Yin, Tie-Yan Liu*

WWW 2021. [[Paper](https://arxiv.org/abs/2102.07372)]

**Knowledge Graph-based Event Embedding Framework for Financial Quantitative Investments**

*Dawei Cheng, Fangzhou Yang, Xiaoyang Wang, Ying Zhang, Liqing Zhang*

SIGIR 2020. [[Paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401427)]


## Reinforcement Learning-based Models

**Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools**

WWW 2024. [[Paper](https://arxiv.org/pdf/2311.10801.pdf)] | [[Code](https://github.com/DVampire/EarnMore)]

**FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition**

*Jeon, Jihyeong and Park, Jiwon and Park, Chanhee and Kang, U*

KDD 2024. [[Paper](https://dl.acm.org/doi/10.1145/3637528.3671668)]

**Asymmetric Graph-Based Deep Reinforcement Learning for Portfolio Optimization**

*Haoyu Sun, Xin Liu, Yuxuan Bian, Peng Zhu, Dawei Cheng, Yuqi Liang*

ECML PKDD 2024. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70378-2_11)]

**NGDRL: A Dynamic News Graph-Based Deep Reinforcement Learning Framework for Portfolio Optimization**

*Yuxuan Bian, Haoyu Sun, Yang Lei, Peng Zhu, Dawei Cheng*

DASFAA 2024. [[Paper](https://link.springer.com/chapter/10.1007/978-981-97-5572-1_29)]

**Efficient Continuous Space Policy Optimization for High-frequency Trading**

*Li Han, Nan Ding, Guoxuan Wang, Dawei Cheng, Yuqi Liang*

KDD 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599813)]

**Optimal Action Space Search: An Effective Deep Reinforcement Learning Method for Algorithmic Trading**

*Zhongjie Duan, Cen Chen, Dawei Cheng, Yuqi Liang, Weining Qian*

CIKM 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557412)] | [[Codes](https://github.com/ECNU-CILAB/OASS)]

**MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading**

*Chuqiao Zong, Chaojie Wang, Molei Qin, Lei Feng, Xinrun Wang, Bo An*

KDD 2024. [[Paper](https://arxiv.org/abs/2406.14537)] | [[Codes](https://github.com/ZONG0004/MacroHFT)]


## Transformer-based Models

**MASTER: Market-Guided Stock Transformer for Stock Price Forecasting**

*Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, Sen Huang*

AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/27767)] | [[Codes](https://github.com/SJTU-DMTai/MASTER)]

**CI-STHPAN: Pre-trained Attention Network for Stock Selection with Channel-Independent Spatio-Temporal Hypergraph**

*Hongjie Xia, Huijie Ao, Long Li, Yu Liu, Sen Liu, Guangnan Ye, Hongfeng Chai*

AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28770)] | [[Codes](https://github.com/Harryx2019/CI-STHPAN)]

**Predicting stock market trends with self-supervised learning**

*Zelin Ying, Dawei Cheng, Cen Chen, Xiang Li, Peng Zhu, Yifeng Luo, Yuqi Liang*

Neurocomputing 2024. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231223011566)]

**Multi-scale Time Based Stock Appreciation Ranking Prediction via Price Co-movement Discrimination**

*Ruyao Xu, Dawei Cheng, Cen Chen, Siqiang Luo, Yifeng Luo, Weining Qian*

DASFAA 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-00129-1_39)] | [[Codes](https://github.com/ECNU-CILAB/MPS)]

**Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport**

*Hengxu Lin, Dong Zhou, Weiqing Liu, Jiang Bian*

KDD 2021. [[Paper](https://arxiv.org/abs/2106.12950)] | [[Codes](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA)]

**Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts**

*Yoo, Jaemin and Soun, Yejun and Park, Yong-chan and Kang, U*

KDD 2021. [[Paper](https://datalab.snu.ac.kr/~ukang/papers/dtmlKDD21.pdf)] | [[Codes](https://github.com/simonjisu/DTML-pytorch)]


## Generative Methods based Models

**DiffsFormer: A Diffusion Transformer on Stock Factor Augmentation**

*Yuan Gao, Haokun Chen, Xiang Wang, Zhicai Wang, Xue Wang, Jinyang Gao, Bolin Ding*

[[Paper](https://arxiv.org/abs/2402.06656)]

**FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns**

*Yitong Duan, Lei Wang, Qizhong Zhang, Jian Li*

AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20369)] | [[Codes](https://github.com/harrishe1999/FactorVAE)]

**GENERATIVE LEARNING FOR FINANCIAL TIME SERIES WITH IRREGULAR AND SCALE-INVARIANT PATTERNS**

*Hongbin Huang, Minghua Chen, and Xiao Qiao*

ICLR 2024. [[Paper](https://openreview.net/forum?id=CdjnzWsQax)]

## Classical Time Series Models

**Learning connections in financial time series**

*Ganeshapillai, Gartheeban, John Guttag, and Andrew Lo.*

ICML 2013. [[Paper](http://proceedings.mlr.press/v28/ganeshapillai13.pdf)] | [[Codes]()]


## Quantitative Open Sourced Framework

**RD-Agent: Autonomous evolving agents for industrial data-drive R&D**

*Microsoft Research Asia*

$2024$. [[Codes](https://github.com/microsoft/RD-Agent)]

**Qlib: An AI-oriented Quantitative Investment Platform**

*Microsoft Research Asia*

$2021$. [[Paper](https://arxiv.org/abs/2009.11189)] | [[Codes](https://github.com/microsoft/qlib)]


## Survey

**Stock Market Prediction via Deep Learning Techniques: A Survey**

*Jinan Zou, Qingying Zhao, Yang Jiao, Haiyao Cao, Yanxi Liu, Qingsen Yan, Ehsan Abbasnejad, Lingqiao Liu, Javen Qinfeng Shi*

$2023$. [[Paper](https://arxiv.org/abs/2212.12717)]



## All Thanks to Our Contributors :

<a href="https://github.com/TongjiFinLab/awesome-financial-time-series-forecasting/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/awesome-financial-time-series-forecasting" />
</a>
