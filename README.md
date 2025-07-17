# A Spiking Neural Network With Adaptive Graph Convolution and LSTM for EEG-Based Brain–Computer Interfaces [[Paper]](https://ieeexplore.ieee.org/document/10049464)

This repository contains the implementation of the following paper:

> *A Spiking Neural Network With Adaptive Graph Convolution and LSTM for EEG-Based Brain–Computer Interfaces*,  
> IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE), 2023.

---

## 🧠 Overview

Electroencephalography (EEG) signals classification is essential for the brain-computer interface (BCI). Recently, energy-efficient spiking neural networks (SNNs) have shown great potential in EEG analysis due to their ability to capture the complex dynamic properties of biological neurons while also processing stimulus information through precisely timed spike trains. However, most existing methods do not effectively mine the specific spatial topology of EEG channels and temporal dependencies of the encoded EEG spikes. Moreover, most are designed for specific BCI tasks and lack some generality. Hence, this study presents a novel SNN model with the customized spike-based adaptive graph convolution and long short-term memory (LSTM), termed SGLNet, for EEG-based BCIs. Specifically, we first adopt a learnable spike encoder to convert the raw EEG signals into spike trains. Then, we tailor the concepts of the multi-head adaptive graph convolution to SNN so that it can make good use of the intrinsic spatial topology information among distinct EEG channels. Finally, we design the spike-based LSTM units to further capture the temporal dependencies of the spikes. We evaluate our proposed model on two publicly available datasets from two representative fields of BCI, notably emotion recognition, and motor imagery decoding. The empirical evaluations demonstrate that SGLNet consistently outperforms existing state-of-the-art EEG classification algorithms. This work provides a new perspective for exploring high-performance SNNs for future BCIs with rich spatiotemporal dynamics.

---

## 📁 Project Structure

```bash
├── model/
│   └── SGLNet.py          # Main model definition
└── README.md
```

## 📌 Citation
Please cite our paper if you use this code in your work:

```
@ARTICLE{10049464,
  author={Gong, Peiliang and Wang, Pengpai and Zhou, Yueying and Zhang, Daoqiang},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={A Spiking Neural Network With Adaptive Graph Convolution and LSTM for EEG-Based Brain-Computer Interfaces}, 
  year={2023},
  volume={31},
  number={},
  pages={1440-1450},
  keywords={Electroencephalography;Brain modeling;Neurons;Convolution;Task analysis;Biological system modeling;Micromechanical devices;EEG classification;brain-computer interface;spiking neural network;graph convolution;LSTM},
  doi={10.1109/TNSRE.2023.3246989}}
```
