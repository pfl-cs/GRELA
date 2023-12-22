# GRELA: Exploiting Graph Representation Learning in Effective Approximate Query Processing 

We propose a Graph REpresentation Learning based model for Approximate query processing. The core idea is to model the aggregate functions and the query predicates as task and query nodes respectively in a graph and learn appropriate node embeddings with GRELA's two modules for them. In particular, the Encoder module coalesce query predicates and underlying data into embeddings for query nodes. The Graph module bridges task nodes and query nodes so that each task node can aggregate the information from its neighborhood into its embedding. 
The estimates of queries with multiple aggregate functions are obtained through the inner products of the corresponding query and task embeddings.

## Requirements

```bash
- Python 3.8+
- Torch 1.9.1+, numpy, scipy, argparse
```

## Dataset

```bash
- cd GRELA 
- tar zxvf data.tar.gz 
```

## First thing to use the code
- You need to make some configurations of the data locations. Run the following scripts.
```bash
- cd GRELA/src
- python init/initialize.py
```

## GRELA Training & Evaluation
- We have provided the STATS dataset, which encompasses two types of workloads: 'static' and 'dynamic'. The static workload comprises only SQL queries, while the dynamic workload includes not just SQL queries but also insert, delete, and update statements. In both workloads, each SQL query is labeled with either a 'training' or 'testing' tag. The training queries, along with their true execution results—also known as ground truth or labels—are divided into training and validation data. This division is done to optimize the parameters of GRELA. On the other hand, the testing queries and their corresponding labels are utilized to assess GRELA's performance.
- The script 'run.py' is designed to train or evaluate GRELA across various workloads. By assigning different values to the command-line arguments, users can control the script to perform a range of functions including  model training and performance evaluation.
- For example, if we want to train GRELA on the static dataset, run the follwing scripts. Note that the first execution will take several more seconds because the histograms and features of the queries in the whole workload need to be built.
```bash
- cd GRELA/src;
- python run.py --train_model True --wl_type static
```
- The following scripts evaluate GRELA on the dynamic workload.
```bash
- cd GRELA/src;
- python run.py --eval_model True --wl_type dynamic
```
