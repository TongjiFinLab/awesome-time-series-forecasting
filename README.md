[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/TongjiFinLab/awesome-financial-time-series-forecasting) 

# Awesome Time Series Forecasting Papers and Codes


+ **Update**: This repository is actively updated.  `2025/05/15`
+ **Collection**: We've compiled a comprehensive list of awesome financial time series forecasting papers and codes.
+ **Collaborate**: If there’s anything missing or if you'd like to contribute, please don't hesitate to get in touch!

## Contents

- [Awesome Financial Time Series Forecasting Papers and Codes](#awesome-financial-time-series-forecasting-papers-and-codes)
  - [Contents](#contents)
  - [A. LLM-based Financial Time Series Forecasting Models](#a-llm-based-financial-time-series-forecasting-models)
  - [B. LLM-based Financial Models](#b-llm-based-financial-models)
    - [AI Agents for Finance](#ai-agents-for-finance)
    - [Stock Prediction and Analysis](#stock-prediction-and-analysis)
    - [Benchmarks and Datasets](#benchmarks-and-datasets)
    - [General Financial Large Language Models](#general-financial-large-language-models)
  - [C. Graph Neural Network-based Models](#c-graph-neural-network-based-models)
  - [D. Reinforcement Learning-based Models](#d-reinforcement-learning-based-models)
  - [E. Transformer-based Models](#e-transformer-based-models)
  - [F. Generative Methods based Models](#f-generative-methods-based-models)
  - [G. Classical Time Series Models](#g-classical-time-series-models)
  - [H. Quantitative Open Sourced Framework](#h-quantitative-open-sourced-framework)
  - [I. Alpha Factor Mining](#i-alpha-factor-mining)
  - [J. Survey](#j-survey)
  - [All Thanks to Our Contributors :](#all-thanks-to-our-contributors-)


## A. LLM-based Financial Time Series Forecasting Models

**AutoTimes: Autoregressive Time Series Forecasters via Large Language Models**<br>
*Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, Mingsheng Long*<br>
NeurIPS 2024. [[Paper](https://arxiv.org/abs/2402.02370)] | [[Codes](https://github.com/thuml/AutoTimes)]

**Are Language Models Actually Useful for Time Series Forecasting?** <br>
*Mingtian Tan, Mike A. Merrill, Vinayak Gupta, Tim Althoff, Thomas Hartvigsen*<br>
NeurIPS 2024. [[Paper](https://arxiv.org/abs/2406.16964)] | [[Codes](https://github.com/BennyTMT/LLMsForTimeSeries)]

**Time-FFM: Towards LM-Empowered Federated Foundation Model for Time Series Forecasting**<br>
*Qingxiang Liu, Xu Liu, Chenghao Liu, Qingsong Wen, Yuxuan Liang* <br>
NeurIPS 2024. [[Paper](https://arxiv.org/abs/2405.14252)]

**TEMPO: PROMPT-BASED GENERATIVE PRE-TRAINED TRANSFORMER FOR TIME SERIES FORECASTING**<br>
*Defu Cao, Furong Jia, Sercan O. Arık, Tomas Pfister, Yixiang Zheng, Wen Ye, Yan Liu* <br>
ICLR 2024. [[Paper](https://openreview.net/forum?id=YH5w12OUuU)] | [[Codes](https://github.com/DC-research/TEMPO)]

**GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting**<br>
*Furong Jia, Kevin Wang, Yixiang Zheng, Defu Cao, Yan Liu*<br>
AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/30383)]

**Multi-Patch Prediction: Adapting Language Models for Time Series Representation Learning**<br>
*Qingxiang Liu, Xu Liu, Chenghao Liu, Qingsong Wen, Yuxuan Liang*<br>
ICML 2024. [[Paper](https://openreview.net/forum?id=Rx9GMufByc)] | [[Codes](https://github.com/yxbian23/aLLM4TS)]

**MOMENT: A Family of Open Time-series Foundation Models**<br>
*Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski*<br>
ICML 2024. [[Paper](https://arxiv.org/abs/2402.03885)] | [[Codes](https://github.com/moment-timeseries-foundation-model/moment)]

**TIME-LLM: TIME SERIES FORECASTING BY REPROGRAMMING LARGE LANGUAGE MODELS**<br>
*Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, et.al.*<br>
ICLR 2024. [[Paper](https://arxiv.org/abs/2310.01728)] | [[Codes](https://github.com/KimMeen/Time-LLM)]

**Timer: Generative Pre-trained Transformers Are Large Time Series Models**<br>
*Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, Mingsheng Long*<br>
ICML 2024. [[Paper](https://arxiv.org/abs/2402.02368)] | [[Codes](https://github.com/thuml/Large-Time-Series-Model)]

**Unified Training of Universal Time Series Forecasting Transformers**<br>
*Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, Doyen Sahoo*<br>
ICML 2024. [[Paper](https://arxiv.org/abs/2402.02592)]


## B. LLM-based Financial Models

### AI Agents for Finance

**LLM-Based Routing in Mixture of Experts: A Novel Framework for Trading**<br>
*Kuan-Ming Liu, Ming-Chih Lo*<br>
AAAI 2025. [[Paper](https://arxiv.org/abs/2501.09636)]

**TradingAgents: Multi-Agents LLM Financial Trading Framework**<br>
*Yijia Xiao, Edward Sun, Di Luo, Wei Wang*<br>
AAAI 2025. [[Paper](https://arxiv.org/abs/2412.20138)]

**Automate Strategy Finding with LLM in Quant investment**<br>
*Zhizhuo Kou, Holam Yu, Jingshu Peng, Lei Chen*<br>
$2024$. [[Paper](https://arxiv.org/abs/2409.06289)]

**FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making**<br>
*Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yupeng Cao, Qianqian Xie, et.al*<br>
NeurIPS 2024. [[Paper](https://arxiv.org/abs/2407.06567)] | [[Codes](https://github.com/The-FinAI/FinCon)]

**FinRobot: AI Agent for Equity Research and Valuation with Large Language Models**<br>
*Tianyu Zhou, Pinqiao Wang, Yilin Wu, Hongyang Yang*<br>
ICAIF 2024. [[Paper](https://arxiv.org/abs/2411.08804)] | [[Codes](https://github.com/AI4Finance-Foundation/FinRobot)]

**A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist**<br>
*Wentao Zhang, Lingxuan Zhao, Haochong Xia, Shuo Sun, Jiaze Sun, Molei Qin, Bo An, et.al*<br>
KDD 2024. [[Paper](https://arxiv.org/abs/2402.18485)]

### Stock Prediction and Analysis

**FinLlama: LLM-Based Financial Sentiment Analysis for Algorithmic Trading**<br>
*Jinyong Fan, Yanyan Shen*<br>
ICAIF 2024. [[Paper](https://dl.acm.org/doi/10.1145/3677052.3698696)]

**StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price Forecasting**<br>
*Jinyong Fan, Yanyan Shen*<br>
AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28681)] | [[Codes](https://github.com/SJTU-DMTai/StockMixer)]

**From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection**<br>
*Xinlei Wang, Maike Feng, Jing Qiu, Jinjin Gu, Junhua Zhao*<br>
NeurIPS 2024. [[Paper](https://arxiv.org/abs/2409.17515)] | [[Codes](https://github.com/ameliawong1996/From_News_to_Forecast)]

**LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction**<br>
*Meiyun Wang, Kiyoshi Izumi, Hiroki Sakaji*<br>
ACL 2024. [[Paper](https://aclanthology.org/2024.findings-acl.185.pdf)]

**Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models**<br>
*Kelvin J.L. Koa, Yunshan Ma,Ritchie Ng, Tat-Seng Chua*<br>
WWW 2024. [[Paper](https://arxiv.org/abs/2402.03659)] | [[Codes](https://github.com/koa-fin/sep)]

**S2IP-LLM: Semantic Space Informed Prompt Learning with LLM forTime Series Forecasting** <br>
*Zijie Pan, Yushan Jiang, Sahil Garg, Anderson Schneider, Yuriy Nevmyvaka, Dongjin Song*<br>
ICML 2024. [[Paper](https://arxiv.org/abs/2403.05798)] | [[Codes](https://github.com/panzijie825/S2IP-LLM)]

### Benchmarks and Datasets

**CFGPT: Chinese Financial Assistant with Large Language Model**<br>
*Jiangtong Li, Yuxuan Bian, Guoxuan Wang, Yang Lei, Dawei Cheng, Zhijun Ding, Changjun Jiang*<br>
$2024$. [[Paper](https://arxiv.org/pdf/2309.10654)] | [[Codes](https://github.com/TongjiFinLab/CFBenchmark)]

**RA-CFGPT: Chinese financial assistant with retrieval-augmented large language model**<br>
*Jiangtong Li, Yang Lei, Yuxuan Bian, Dawei Cheng, Zhijun Ding, Changjun Jiang*<br>
FCS 2024. [[Paper](https://link.springer.com/article/10.1007/s11704-024-31018-5)]

**CSPRD: A Financial Policy Retrieval Dataset for Chinese Stock Market**<br>
*Jinyuan Wang, Zhong Wang, Zeyang Zhu, Jinhao Xie, Yong Yu, Yongjian Fei, Yue Huang, Dawei Cheng, Hai Zhao*<br>
DEXA 2024. [[Paper](https://arxiv.org/abs/2309.04389)] | [[Codes](https://github.com/noewangjy/csprd_dataset)]

**FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets**<br>
*Neng Wang, Hongyang Yang, Christina Dan Wang*<br>
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.04793)] | [[Codes](https://github.com/AI4Finance-Foundation/FinGPT)]

**FinGPT: Democratizing Internet-scale Data for Financial Large Language Models**<br>
*Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, Daochen Zha*<br>
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2307.10485)] | [[Codes](https://github.com/AI4Finance-Foundation/FinGPT)]

### General Financial Large Language Models

**BloombergGPT: A Large Language Model for Finance**<br>
*Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Gideon Mann, et.al*<br>
$2023$. [[Paper](https://arxiv.org/abs/2303.17564)]


## C. Graph Neural Network-based Models

**LSR-IGRU: Stock Trend Prediction Based on Long Short-Term Relationships and Improved GRU**<br>
*Peng Zhu, Yuante Li, Yifan Hu, Qinyuan Liu, Dawei Cheng, Yuqi Liang*<br>
CIKM 2024. [[Paper](https://arxiv.org/abs/2409.08282)] | [[Codes](https://github.com/ZP1481616577/Baselines_LSR-IGRU)]

**Automatic De-Biased Temporal-Relational Modeling for Stock Investment Recommendation**<br>
*Weijun Chen, Shun Li, Xipu Yu, Heyuan Wang, Wei Chen, Tengjiao Wang*<br>
IJCAI 2024. [[Paper](https://www.ijcai.org/proceedings/2024/221)]

**MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction**<br>
*Hao Qian, Hongting Zhou, Qian Zhao, Hao Chen, Hongxiang Yao, Jingwei Wang, Ziqi Liu, Fei Yu, Zhiqiang Zhang, Jun Zhou*<br>
AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29381)]

**ECHO-GL: Earnings Calls-Driven Heterogeneous Graph Learning for Stock Movement Prediction**<br>
*Mengpu Liu, Mengying Zhu, Xiuyuan Wang, Guofang Ma, Jianwei Yin, Xiaolin Zheng*<br>
AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29305)] | [[Codes](https://github.com/pupu0302/ECHOGL)]

**TCGPN: Temporal-Correlation Graph Pre-trained Network for Stock Forecasting**<br>
*Wenbo Yan, Ying Tan*<br>
[[Paper](https://arxiv.org/abs/2407.18519)]

**Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction**<br>
*Sheng Xiang, Dawei Cheng, Chencheng Shang, Ying Zhang, Yuqi Liang*<br>
CIKM 2022. [[Paper](https://arxiv.org/abs/2305.08740)] | [[Codes](https://github.com/finint/THGNN)]

**Relational Temporal Graph Convolutional Networks for Ranking-Based Stock Prediction**<br>
*Zetao Zheng, Jie Shao, Jia Zhu, Heng Tao Shen*<br>
ICDE 2023. [[Paper](https://ieeexplore.ieee.org/document/10184655)] | [[Codes](https://github.com/zhengzetao/RTGCN)]

**Temporal-Relational hypergraph tri-Attention networks for stock trend prediction**<br>
*Chaoran Cui, Xiaojie Li, Chunyun Zhang, Weili Guan, Meng Wang*<br>
Pattern Recognition 2023. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323004570)] | [[Codes](https://github.com/lixiaojieff/HGTAN)]

**Financial time series forecasting with multi-modality graph neural network**<br>
*Dawei Cheng, Fangzhou Yang, Sheng Xiang, Jin Liu*<br>
Pattern Recognition 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S003132032100399X)] | [[Codes](https://github.com/finint/MAGNN)]

**Hierarchical Adaptive Temporal-Relational Modeling for Stock Trend Prediction**<br>
*Heyuan Wang, Shun Li, Tengjiao Wang, Jiayi Zheng*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/508)] | [[Codes](https://github.com/lixiaojieff/HGTAN)]

**REST: Relational Event-driven Stock Trend Forecasting**<br>
*Wentao Xu, Weiqing Liu, Chang Xu, Jiang Bian, Jian Yin, Tie-Yan Liu*<br>
WWW 2021. [[Paper](https://arxiv.org/abs/2102.07372)]

**Knowledge Graph-based Event Embedding Framework for Financial Quantitative Investments**<br>
*Dawei Cheng, Fangzhou Yang, Xiaoyang Wang, Ying Zhang, Liqing Zhang*<br>
SIGIR 2020. [[Paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401427)]


## D. Reinforcement Learning-based Models

**MacMic: Executing Iceberg Orders via Hierarchical Reinforcement Learning**<br>
*Hui Niu, Siyuan Li, Jian Li*<br>
IJCAI 2024. [[Paper](https://www.ijcai.org/proceedings/2024/0664.pdf)]

**Cross-contextual Sequential Optimization via Deep Reinforcement Learning for Algorithmic Trading**<br>
*Kaiming Pan, Yifan Hu, Li Han, Haoyu Sun, Dawei Cheng, Yuqi Liang*<br>
CIKM 2024. [[Paper](https://dl.acm.org/doi/10.1145/3627673.3680101)]

**Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools**<br>
*Wentao Zhang, Yilei Zhao, Shuo Sun, Jie Ying, Yonggang Xie, Zitao Song, Xinrun Wang, Bo An*<br>
WWW 2024. [[Paper](https://arxiv.org/pdf/2311.10801.pdf)] | [[Codes](https://github.com/DVampire/EarnMore)]<br>

**FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition**<br>
*Jeon, Jihyeong and Park, Jiwon and Park, Chanhee and Kang, U*<br>
KDD 2024. [[Paper](https://dl.acm.org/doi/10.1145/3637528.3671668)]

**MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading**<br>
*Chuqiao Zong, Chaojie Wang, Molei Qin, Lei Feng, Xinrun Wang, Bo An*<br>
KDD 2024. [[Paper](https://arxiv.org/abs/2406.14537)] | [[Codes](https://github.com/ZONG0004/MacroHFT)]

**Asymmetric Graph-Based Deep Reinforcement Learning for Portfolio Optimization**<br>
*Haoyu Sun, Xin Liu, Yuxuan Bian, Peng Zhu, Dawei Cheng, Yuqi Liang*<br>
ECML PKDD 2024. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70378-2_11)]

**NGDRL: A Dynamic News Graph-Based Deep Reinforcement Learning Framework for Portfolio Optimization**<br>
*Yuxuan Bian, Haoyu Sun, Yang Lei, Peng Zhu, Dawei Cheng*<br>
DASFAA 2024. [[Paper](https://link.springer.com/chapter/10.1007/978-981-97-5572-1_29)]

**Efficient Continuous Space Policy Optimization for High-frequency Trading**<br>
*Li Han, Nan Ding, Guoxuan Wang, Dawei Cheng, Yuqi Liang*<br>
KDD 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599813)]

**Optimal Action Space Search: An Effective Deep Reinforcement Learning Method for Algorithmic Trading**<br>
*Zhongjie Duan, Cen Chen, Dawei Cheng, Yuqi Liang, Weining Qian*<br>
CIKM 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557412)] | [[Codes](https://github.com/ECNU-CILAB/OASS)]


## E. Transformer-based Models

**MASTER: Market-Guided Stock Transformer for Stock Price Forecasting**<br>
*Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, Sen Huang*<br>
AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/27767)] | [[Codes](https://github.com/SJTU-DMTai/MASTER)]

**CI-STHPAN: Pre-trained Attention Network for Stock Selection with Channel-Independent Spatio-Temporal Hypergraph**<br>
*Hongjie Xia, Huijie Ao, Long Li, Yu Liu, Sen Liu, Guangnan Ye, Hongfeng Chai*<br>
AAAI 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28770)] | [[Codes](https://github.com/Harryx2019/CI-STHPAN)]

**Predicting stock market trends with self-supervised learning**<br>
*Zelin Ying, Dawei Cheng, Cen Chen, Xiang Li, Peng Zhu, Yifeng Luo, Yuqi Liang*<br>
Neurocomputing 2024. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231223011566)]

**Multi-scale Time Based Stock Appreciation Ranking Prediction via Price Co-movement Discrimination**<br>
*Ruyao Xu, Dawei Cheng, Cen Chen, Siqiang Luo, Yifeng Luo, Weining Qian*<br>

DASFAA 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-00129-1_39)] | [[Codes](https://github.com/ECNU-CILAB/MPS)]

**Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport**<br>
*Hengxu Lin, Dong Zhou, Weiqing Liu, Jiang Bian*<br>
KDD 2021. [[Paper](https://arxiv.org/abs/2106.12950)] | [[Codes](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA)]

**Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts**<br>
*Yoo, Jaemin and Soun, Yejun and Park, Yong-chan and Kang, U*<br>
KDD 2021. [[Paper](https://datalab.snu.ac.kr/~ukang/papers/dtmlKDD21.pdf)] | [[Codes](https://github.com/simonjisu/DTML-pytorch)]


## F. Generative Methods based Models

**DHMoE: Diffusion Generated Hierarchical Multi-Granular Expertise for Stock Prediction**<br>
*Weijun Chen, Yanze Wang*<br>
AAAI 2025. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33250)]

**Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context**<br>
*Haochong Xia, Shuo Sun, Xinrun Wang, Bo An*<br>
AAAI 2024. [[Paper](https://arxiv.org/abs/2309.07708)] | [[Codes](https://github.com/XiaHaochong98/Market-GAN)]

**RSAP-DFM: Regime-Shifting Adaptive Posterior Dynamic Factor Model for Stock Returns Prediction**<br>
*Quanzhou Xiang, Zhan Chen, Qi Sun, Rujun Jiang*<br>
IJCAI 2024. [[Paper](https://www.ijcai.org/proceedings/2024/0676.pdf)]

**Automatic De-Biased Temporal-Relational Modeling for Stock Investment Recommendation**<br>
*Weijun Chen, Shun Li, Xipu Yu, Heyuan Wang, Wei Chen, Tengjiao Wang*<br>
IJCAI 2024. [[Paper](https://www.ijcai.org/proceedings/2024/0221.pdf)]

**GENERATIVE LEARNING FOR FINANCIAL TIME SERIES WITH IRREGULAR AND SCALE-INVARIANT PATTERNS**<br>
*Hongbin Huang, Minghua Chen, and Xiao Qiao*<br>
ICLR 2024. [[Paper](https://openreview.net/forum?id=CdjnzWsQax)]

**DiffsFormer: A Diffusion Transformer on Stock Factor Augmentation**<br>
*Yuan Gao, Haokun Chen, Xiang Wang, Zhicai Wang, Xue Wang, Jinyang Gao, Bolin Ding*<br>
$2024$. [[Paper](https://arxiv.org/abs/2402.06656)]

**FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns**<br>
*Yitong Duan, Lei Wang, Qizhong Zhang, Jian Li*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20369)] | [[Codes](https://github.com/harrishe1999/FactorVAE)]

## G. Classical Time Series Models

**Learning connections in financial time series**<br>
*Ganeshapillai, Gartheeban, John Guttag, and Andrew Lo.*<br>
ICML 2013. [[Paper](http://proceedings.mlr.press/v28/ganeshapillai13.pdf)] | [[Codes]()]


## H. Quantitative Open Sourced Framework

**RD-Agent: Autonomous evolving agents for industrial data-drive R&D**<br>
*Microsoft Research Asia*<br>
$2024$. [[Codes](https://github.com/microsoft/RD-Agent)]

**Qlib: An AI-oriented Quantitative Investment Platform**<br>
*Microsoft Research Asia*<br>
$2021$. [[Paper](https://arxiv.org/abs/2009.11189)] | [[Codes](https://github.com/microsoft/qlib)]


## I. Alpha Factor Mining
**AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors**<br>
*Hao Shi, Weili Song, Xinting Zhang, Jiahe Shi, Cuicui Luo, Xiang Ao, Hamid Arian, Luis Seco*<br>
AAAI 2025. [[Paper](https://arxiv.org/abs/2406.18394)] | [[Codes](https://github.com/DulyHao/AlphaForge)]

## J. Survey

**From Deep Learning to LLMs: A survey of AI in Quantitative Investment**<br>
*Bokai Cao, Saizhuo Wang, Xinyi Lin, Xiaojun Wu, Haohan Zhang, Lionel M Ni, Jian Guo*<br>
$2025$. [[Paper](https://arxiv.org/pdf/2503.21422?)]

**Large Language Model Agent in Financial Trading: A Survey**<br>
*Han Ding, Yinheng Li, Junhao Wang, Hang Chen*<br>
$2024$. [[Paper](https://arxiv.org/abs/2408.06361)]

**Stock Market Prediction via Deep Learning Techniques: A Survey**<br>
*Jinan Zou, Qingying Zhao, Yang Jiao, Haiyao Cao, Yanxi Liu, Qingsen Yan, Ehsan Abbasnejad, Lingqiao Liu, Javen Qinfeng Shi*<br>
$2023$. [[Paper](https://arxiv.org/abs/2212.12717)]



## All Thanks to Our Contributors :

<a href="https://github.com/TongjiFinLab/awesome-financial-time-series-forecasting/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/awesome-financial-time-series-forecasting" />
</a>
