# Deep RL Recommender System Project for E-commerce (AIPI531)

## Team Members:
Elisa Chen, Beibei Du, Aditya John, Medha Sreenivasan[in alphabetical order by last name]

<img width="794" alt="Screen Shot 2023-04-20 at 6 59 18 PM" src="https://user-images.githubusercontent.com/60382493/233504139-a65d59c8-ee63-4e43-abed-958199c858ab.png">

## Motivation
The rapid expansion of the e-commerce industry has led to an overwhelming amount of choices for consumers, making it increasingly difficult for customers to find relevant products that fullfill their preferences and needs. To solve this issue, personalized product recommendations have become an important aspect of enhancing user experience and ensuring customer satisfaction. Most importantly, increasing the sales and prestige of the companies to achieve double win. As a result, there is a growing demand for innovative and effective product recommendation systems that can adapt to users' preferences in real-time and provide accurate recommendations. 

## Goal
This project aims to explore recommender systems by implementing algorithms into datasets, specifically focusing on session-based and sequential recommendation techniques that can be utilized in e-commerce use cases. By experimenting with different methods, including Deep Reinforcement Learning (DRL), we strive to improve the performance of recommendation systems and ultimately contribute to an enhanced user experience in the e-commerce domain. In this project, we are comparing the performance of a vanilla GRU model to a hRNN (hierarchical RNN) model that also accounts for item features. Introducing item features as part of the recommendation system can be greatly beneficial in cold start situations.


## Related Works - FILL IN
- Mention the papers
hRNN models: https://openreview.net/pdf?id=8QFKbygVy4r
Inspiration: https://arxiv.org/pdf/2111.03474.pdf 

## Data

We have used two E-commerce datasets for our project. Following are the details of the datasets used

**Dataset #1 - Retail Rocket Dataset**

The first dataset used was from Retail Rocket. Retail Rocket is a company that generates personalized product recommendations for shopping websites and provides customer segmentation based on user interests and other parameters. The dataset was collected from a real-world e-commerce website and consisted of raw data, i.e., data without any content transformation. However, all values are hashed to address confidentiality concerns. Among the files in the dataset, only the behavior data (`events.csv`) is used in this project. The behavior data is a timestamped log of events like clicks, add to carts, and transactions that represent different interactions made by visitors on the e-commerce website over a time period of 4.5 months. There are a total of 2756101 events produced by 1407580 unique visitors. We also leveraged the `item_features_x.csv` dataset to capture the features of each item. The file contains information about the properties and values of each item. 

**Dataset #2 - H&M Dataset**

The second dataset was from H&M Group. H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Their online store offers shoppers an extensive selection of products to browse through. The available metadata spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images. Among the files in the dataset, we used the `transactions_train.csv` for our events data and `articles.csv` data for item features. This dataset contains information about the properties of each item. 

## Methodology
### One Hot Encoding of Item Features
For both Retail Rocket and H&M datasets, we had to perform feature selection for the most pertinent properties and one-hot-encode all the values per each item. For this analysis, we only considered the 2000 most frequent properties for the retail rocket dataset and XXX most frequent properties for the H&M dataset. On average, the top N properties covered ~50% of all possible properties in the feature space. 

### Re-defined Loss Function
To create a hRNN model, we had to pass in the item feature matrix through another feed-forward layer to create item embedding vectors that can be incorporated as part of the score for the jth item. We modified the existing loss function in the `SNQN.py` file with the new loss function that also accounts for item embedding vectors.

## Instructions For Running The Code

**Train Model Using Retail Rocket Dataset - DOUBLE CHECK THE PATH**

Open `20_code/RR/AIPI531_Project_RR.ipynb` notebook in Google Colab instance. This file contains all code to reproduce the results for Retail Rocket Dataset. Running the notebook should train and evaluate the model on purchase and click hr & ndcg.

**Train Model Using H&M Dataset - DOUBLE CHECK THE PATH**

Open `20_code/RR/AIPI531_Project_HM.ipynb` notebook in Google Colab instance. This file contains all code to reproduce the results for Retail Rocket Dataset. Running the notebook should train and evaluate the model on purchase and click hr & ndcg.

## Results & Discussion

The evaluation metrics used are Click Hit Ratio (HR), Click Normalized Discounted Cumulative Gain (NDCG), Purchase Hit Ratio and Purchase Normalized Discounted Cumulative Gain. 

- NDCG@k measures the quality of the recommendation list based on the top-k ranking of items in the list higher ranked items are scored higher
- HR@k measures whether the ground-truth item is in the top-k positions of the recommendation list generated by the model.

Below are the results for GRU (Baseline) and hRNN (GRU + Iten Features) for our evaluation metrics where NG is short for NDCG for 5 epochs:

### Retail Rocket

**Clicks**

| Model | HR@5   | NG@5   | HR@5 | NG@5 | HR@10 | NG@10 | HR@20 | NG@20 |
|-------|--------|--------|------|------|-------|-------|-------|-------|
| GRU   | 0.0023 | 0.0023 |      |      |       |       |       |       |
| hRNN  | 0.1925 | 0.1435 | 0.2437 | 0.1601 | 0.2721 | 0.1676 | 0.2920 | 0.1723|

**Purchase**

| Model | HR@5   | NG@5   | HR@10 | NG@10 | HR@15 | NG@15 | HR@20 | NG@20 |
|-------|--------|--------|------|------|-------|-------|-------|-------|
| GRU   | 0.0023 | 0.0023 |      |      |       |       |       |       |
| hRNN  | 0.3973 | 0.3122 | 0.4547 | 0.3308 | 0.4880 | 0.3396 | 0.5112 | 0.3451|

### H&M

**Clicks**

| Model | HR@5   | NG@5   | HR@5 | NG@5 | HR@10 | NG@10 | HR@20 | NG@20 |
|-------|--------|--------|------|------|-------|-------|-------|-------|
| GRU   | 0.0023 | 0.0023 |      |      |       |       |       |       |
| hRNN  |        |        |      |      |       |       |       |       |

**Purchase**

| Model | HR@5   | NG@5   | HR@5 | NG@5 | HR@10 | NG@10 | HR@20 | NG@20 |
|-------|--------|--------|------|------|-------|-------|-------|-------|
| GRU   | 0.0023 | 0.0023 |      |      |       |       |       |       |
| hRNN  |        |        |      |      |       |       |       |       |


As we can observe, hRNN performs consistently better ....

## Future Research

The following suggestions could be implemented in a future study for a better model performance:
1) **Hyperparameter Tuning Using Random Search**: The key hyperparameter values impacting the model output are `learning rate`, `epoch number` and `lambda`. We are using 0.005, 5 and 0.15 respectively for our model training. A random search could be used to better optimize the hyperparameter values in the future.

2) **Alternative Evaluation Metric:** We are currently using HR and NDCG to evaluate the model. However, there are other metrics such as MRR and MAP that could also be used to evaluate the model performance. Using alternative metrics could help us gain a more nuanced and encompassing understanding of the model performance. 

## Contribution (Task A: CQL loss; Task B: Item/User Features)
- Elisa Chen: 1) Incorporating item features as part of source code (SNQN.py) 2) model training & evaluation.
- Beibei Du: 
- Aditya John: 1) Incorporating item features as part of source code (SNQN.py) 2) model training & evaluation.
- Medha Sreenivasan:



## Reference
[1] Xin, Xin, et al. "Self-supervised reinforcement learning for recommender systems." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020.


## Quick Links to the resources
- Setup Video: https://duke.zoom.us/rec/play/YLp4O2k-92fKFY0sqP8WgwcbtgcZ58yl7EFms3ef8QvoFfNFToUWGsGvH9XyzNRRSfMHN6_EGiUWHaWK.GclYbUf34PNRPJpS?canPlayFromShare=true&from=share_recording_detail&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Fduke.zoom.us%2Frec%2Fshare%2F3qVUTlyJP855r9nNiSEQ2rtTho--Dzo0MjTtB8BMw0CyVXKqAsgxKHo8nzakvT2y.arWt-MPztCNOBMeC
- Hint 2: https://sakai.duke.edu/portal/site/AIPI-531-01-Sp23/tool/1e88ff74-8322-4f03-b00d-5f2fc168b0be?panel=Main
- Hint 1: https://sakai.duke.edu/portal/site/AIPI-531-01-Sp23/tool/1e88ff74-8322-4f03-b00d-5f2fc168b0be?panel=Main
- Setup Source Code: https://colab.research.google.com/drive/1g8JpuF5GoumflDJtMV2uMUfRjY_xheQX?usp=sharing



