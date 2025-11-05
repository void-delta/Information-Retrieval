
# Information Retrieval

I am tasked with building an Information Retrieval system using the Boolean Retrieval method and the Learn-to-Rank algorithm. This system is built to interact with the Cranfield Dataset. 

The Cranfield dataset is a small, historically significant collection of scientific abstracts used to evaluate early information retrieval (IR) systems. Created during the Cranfield experiments in the 1960s, it provided a standard benchmark for assessing IR system performance. Though too small for modern applications, it is still used for teaching, testing baseline models, and exploring the history of IR. 

Digant Singh [@void-delta](https://www.github.com/void-delta)
# Cranfield - Boolean Shortlist & XGBoost LTR Ranking

Built an end-to-end Information Retrieval system using the Cranfield dataset, combining Boolean retrieval for candidate generation and XGBoost (LambdaMART) for learning-to-rank. Implemented an inverted index, feature engineering for query–document pairs, and evaluated ranking performance using metrics like Precision@k, MAP, and nDCG. Deployed an interactive Streamlit interface for querying and visualizing ranked results.
## How to Run the Code
To start everything off, we need to install all the requirements
```bash
  python -m pip install -r requirements.txt
  python -m nltk.downloader stopwords punkt
```
Then we need to build the inverted indices for the system and train the XGBoost Model
```bash
  python ./train.py
```
Further we need to run the evaluation for the Precision@k, MAP and nDCG. For this specific script, you will see the results for n = 10 for baseline metrics
```bash
  python ./evaluate.py
```
Finally we host the entire dataset on Streamlit for interfacing with it and actually using the system we have built
```bash
  python -m streamlit run ./app.py
```

Or you can visit the hosted project on streamlit at https://cranfield-boolean-ltr.streamlit.app/ 

## Results

Following are the results for the Cranfield Dataset using Boolean Shortlisting and Learn-to-Rank algorithm. 

Based on the test split of 20% of the Cranfield Dataset
<img width="573" height="277" alt="image" src="https://github.com/user-attachments/assets/6f0e519a-6462-4bb6-8158-bd21c05615e0" />

We can see that the nDCG value is 1 which shows that the LTR model works very well on this dataset. This might be due to the fact that the dataset itself is very small and the qrel given for the standard queries are very good in training the XGBoost ensemble model. In any case they do very well in the standard queries and we do not have the qrels for any other query that we may want to test.


You can also run the following queries to judge the system's Boolean Shortlising and the XGBoost LTR retrieval.

```bash
  thermodynamic jet propulsion
```
```bash
  aerodynamic streamlining
```

### General Information
If you have any feedback, please reach out to me at
22bcs037@iiitdwd.ac.in


Assignment for CS468: Information Retrieval under Dr. Krishnendu Ghosh.



