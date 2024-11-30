# TH-RotatE: Knowledge Graph-based Fault Diagnosis System for Railway Operational Equipment

This project presents a fault diagnosis system for Chinese railway operational equipment based on a knowledge graph embedding model, TH-RotatE. The model integrates the strengths of TransH and RotatE models for accurate fault diagnosis by leveraging both semantic and structural information from the railway fault dataset.

## Research Problem
The goal of this project is to diagnose faults in Chinese railway operational equipment based on historical fault reports. The dataset includes fault descriptions, causes, and related entities, which are used to build a knowledge graph for fault prediction.

## Approach
We propose an embedding-based model called TH-RotatE, which combines the TransH and RotatE models. The model uses entity and relation embeddings to predict fault relationships, improving diagnostic accuracy.


## Data

The dataset used in this project consists of Chinese railway operational equipment fault reports, covering the period from January 1, 2018, to April 30, 2020. This dataset includes information such as fault phenomena, causes, and fault measures. 

### Data Format
The raw data is in a `.txt` format, and after preprocessing, it is stored in `.json` format for ease of use with the model.

### Data Access
Due to confidentiality restrictions, the dataset is not publicly available. You can contact the data application department of the China Railway Nanning Bureau Group Co., Ltd. at [nanningkyd@163.com](mailto:nanningkyd@163.com) for access to the dataset.


## Usage

### Training the Model
To train the model, run the following command:
```bash
python train.py

## Results

The model's performance is evaluated using metrics such as Mean Reciprocal Rank (MRR), Hit@1, Hit@3, and Hit@10.

Sample results:
- MRR: 0.821
- Hit@1: 0.79
- Hit@3: 0.91
- Hit@10: 0.98

The model achieves high accuracy across various fault types in the Chinese railway operational equipment dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Data Citation

Please cite the dataset as follows:
- **China Railway Nanning Bureau Group Co., Ltd.** (2018-2020). Chinese railway operational equipment fault reports.
