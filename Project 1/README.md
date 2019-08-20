# Topic Modelling on Encrpyted Twitter Dataset
Using Latent Dirichlet Allocation (LDA) for Topic Modelling on Encrypted Twitter Dataset

# Step-by-step
## Step 1: Crawling Twitter Dataset
Getting tweet documents using Twitter API. The results are as follows.

|   | username        | created_at          | text                                        |
|---|-----------------|---------------------|---------------------------------------------|
| 0 | JustNowBulletin | 2019-08-20 14:20:11 | Udacity AWS DeepRacer Scholarship Challenge |
| 1 | sustaintrain    | 2019-08-20 14:15:10 | A new Udacity Green IT training course      |
| 2 | Upwork          | 2019-08-20 13:20:49 | Back to School Isn’t Just For Kids          |

## Step 2: Exploratory Data Analysis
We can visualizing the number of tweets for each day.
![EDA](https://github.com/yasirabd/SPAI/blob/master/Project%201/assets/eda.png "EDA")

## Step 3: Implement Encrpyted Dataset
Encrypt our Twitter dataset.
```python
# initialize EncryptedDataset class
db = EncryptedDataset(bob, alice, secure_worker, max_val_len=280)

# add Twitter dataset into db
for index, row in df_tweet.iterrows():
    # we can make the key combining index and username
    key = str(index) + '@' + row['username']
    values = row['text']
    db.add_entry(key, values)
    
# load encrypted dataset
db.query("0@JustNowBulletin")
# the result: 'udacity aws deepracer scholarship challenge for international students '
```

## Step 4: Topic Modelling
Implementing Latent Dirichlet Allocation on Encrypted Twitter Dataset. The results are as follows.
![LDA](https://github.com/yasirabd/SPAI/blob/master/Project%201/assets/lda.PNG "LDA")
