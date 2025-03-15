# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:34:45 2025

@author: Connor Steward
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import fitz
import re
import spacy
from collections import Counter
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")
# MGTF 423 Data Science Final Project Sentiment Analysis and Correlation with Stock Prices

#Functions used

#Read the pdfs
#Returns a string object that captures from the start header to the end header
#making sure that if a section is over multiple pages it is all captured
def pdf_reader(pdf_path, start_header, end_header):
    doc = fitz.open(pdf_path)
    text = ""
    capture = False
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page = page.get_text()
        lines = page.split('\n')
        for line in lines:
            if re.search(start_header, line, re.IGNORECASE):
                capture = True
                continue
            if re.search(end_header, line, re.IGNORECASE):
                capture = False
                break
            if capture:
                text += line + '\n'
    return text


#Returns a string object from a pdf
def pdf_reader2(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page = page.get_text()
        text += page    
    return text


#Returns a dataframe with the most common words, 
#ignoring common words such as excluded words
def most_common_words(file):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # Words to exclude
    exclude_words = set(['be', 'am', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might',
                     'must', 'can', 'could', 'we', 'our', 'us', 'i', 'me', 'my', 'mine', 'you', 
                     'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 
                     'they', 'them', 'their', 'theirs', 'this', 'that', 'these', 'those', "financial", "our business"])
    noun_chunks = [token.lemma_.lower() for token in doc 
         if token.pos_ == "NOUN" and token.lemma_.lower() not in exclude_words]
    noun_chunk_counts = Counter(noun_chunks)
    verbs = [token.lemma_.lower() for token in doc 
         if token.pos_ == "VERB" and token.lemma_.lower() not in exclude_words]
    verb_counts = Counter(verbs)
    adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
    adjective_counts = Counter(adjectives)
    top_noun_chunks = noun_chunk_counts.most_common(5)
    top_verbs = verb_counts.most_common(5)
    top_adjectives = adjective_counts.most_common(5)
    df_noun_chunks = pd.DataFrame(top_noun_chunks, columns=['item', 'count'])
    df_noun_chunks['type'] = 'noun_chunk'
    df_verbs = pd.DataFrame(top_verbs, columns=['item', 'count'])
    df_verbs['type'] = 'verb'
    df_adjectives = pd.DataFrame(top_adjectives, columns=['item', 'count'])
    df_adjectives['type'] = 'adjective'
    df_combined = pd.concat([df_noun_chunks, df_verbs, df_adjectives], ignore_index=True)
    df_combined = df_combined.sort_values('count', ascending=False)
    return df_combined

# Function to perform sentiment analysis using VADER
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment


def plot_top_words(df, title):
    plt.figure(figsize=(12, 8))
    noun_chunks = df[df['type'] == 'noun_chunk']
    plt.bar(noun_chunks['item'], noun_chunks['count'], label='Noun Chunks', color='blue', alpha=0.7)
    verbs = df[df['type'] == 'verb']
    plt.bar(verbs['item'], verbs['count'], label='Verbs', color='green', alpha=0.7)
    adjectives = df[df['type'] == 'adjective']
    plt.bar(adjectives['item'], adjectives['count'], label='Adjectives', color='orange', alpha=0.7)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Load in all data (pdfs of 10K and 10Qs, stock price data, dates of earning calls)

years = [2024, 2023, 2022]
file_dict = {}
for year in years:
    key = f"K10_{str(year)[-2:]}"
    file_dict[key] = f"IONQ {year} 10K.pdf"

file_dict2 = {}    
for year in years:    
    for i in range(1,4):
        key = f"Q10_{str(year)[-2:]} {i}"
        file_dict2[key] = f"IONQ {year} 10Q{i}.pdf"
        
Ionq = pd.read_csv("IONQ Stock Data.csv")
Ionq["date"] = pd.to_datetime(Ionq["date"])
Ionq.set_index("date", inplace = True)
K10_dates = pd.to_datetime(["03/25/2021", "03/28/2022", "03/30/2023", "02/28/2024"])
Q10_dates = pd.to_datetime(["06/04/2021", "08/16/2021", "11/15/2021",
             "05/16/2022", "08/15/2022", "11/14/2022",
             "05/11/2023", "08/10/2023", "11/09/2023",
             "05/10/2024", "08/07/2024", "11/06/2024"])

# Risks Related to Our Financial Condition and Status as an Early-Stage Company
#Risks Related to Our Business and Industry
# 10Ks have "Our Forward-Looking Roadmap" and 10qs have "Business and Technical Highlights"
# as well as "Impact of the Macroeconomic Climate on Our Business" for both
keywords = ["Our Forward-Looking Roadmap", "Business and Technical Highlights", "Impact of Macroeconomic Climate", 
            "Remaining Challenges in Quantum Computing Evolution"]

# Run on all documents
files = []
for key in file_dict:
    pdf_path = file_dict[key]
    globals()[key] = pdf_reader(pdf_path, keywords[0], "Our Business Model")
    files.append(globals()[key])

files2 = [] 
for key in file_dict2:
    pdf_path = file_dict2[key]
    globals()[key] = pdf_reader(pdf_path, "Business and Technical Highlights", "Key Components of Results of Operations")
    files2.append(globals()[key])
    
# Returning positive aspect of sentiment analysis 
Senti_10K_vs_pos = pd.DataFrame()
i=1
for text in files:
    sentiment = pd.DataFrame([analyze_sentiment_vader(text)['pos'],K10_dates[-i]]).T
    Senti_10K_vs_pos = pd.concat([Senti_10K_vs_pos, sentiment], ignore_index=True)
    i = i+1

#Returning Negative aspect of sentiment analysis 
Senti_10K_vs_neg = pd.DataFrame()
i=1
for text in files:
    sentiment = pd.DataFrame([analyze_sentiment_vader(text)['neg'],K10_dates[-i]]).T
    Senti_10K_vs_neg = pd.concat([Senti_10K_vs_neg, sentiment], ignore_index=True)
    i = i+1

# Connecting dates to each sentiment analysis
Senti_10q_pos = pd.DataFrame()
i=-3
for text in files2:
    sentiment = pd.DataFrame([analyze_sentiment_vader(text)['pos'],Q10_dates[i]]).T
    Senti_10q_pos = pd.concat([Senti_10q_pos, sentiment], ignore_index=True)
    i = i+1
    if i == 0:
        i = -6
    elif i == -3:
        i = -9

Senti_10q_neg = pd.DataFrame()
i=-3
for text in files2:
    sentiment = pd.DataFrame([analyze_sentiment_vader(text)['neg'],Q10_dates[i]]).T
    Senti_10q_neg = pd.concat([Senti_10q_neg, sentiment], ignore_index=True)
    i = i+1
    if i == 0:
        i = -6
    elif i == -3:
        i = -9
        
#Connecting 10K and 10q data
Senti_10K_vs_neg.reset_index(inplace=True, drop =True)
Senti_10K_vs_neg.set_index(1, inplace = True)
Senti_10K_vs_pos.reset_index(inplace=True, drop =True)
Senti_10K_vs_pos.set_index(1, inplace = True)
Senti_10q_neg.reset_index(inplace=True, drop =True)
Senti_10q_neg.set_index(1, inplace = True)
Senti_10q_pos.reset_index(inplace=True, drop =True)
Senti_10q_pos.set_index(1, inplace = True)

senti_pos = pd.concat([Senti_10K_vs_pos, Senti_10q_pos])
senti_pos.sort_index(inplace = True)
senti_neg = pd.concat([Senti_10K_vs_neg, Senti_10q_neg])
senti_neg.sort_index(inplace = True)

#Get stock data
ret_list = []
for index in senti_pos.index:
    d = Ionq["RET"].loc[index]
    ret_list.append({'index': index, 'RET': d})
    
ret = pd.DataFrame(ret_list)
ret.set_index('index', inplace=True)
senti_pos = pd.concat([senti_pos,ret], axis = 1)
senti_pos.sort_index(inplace = True)
senti_neg = pd.concat([senti_neg,ret], axis = 1)
senti_neg.sort_index(inplace = True)


#Run OLS on both positive and negative sentiments to see what beta, correlation, and R2
senti_pos[0] = pd.to_numeric(senti_pos[0], errors='coerce')
senti_pos["RET"] = pd.to_numeric(senti_pos["RET"], errors='coerce')
posOLS = sm.OLS(senti_pos["RET"], sm.add_constant(senti_pos[0])).fit()
pos_corr = senti_pos[0].corr(senti_pos["RET"])
print(f"""The positive regression has an r-squared of {round(posOLS.rsquared,4)}, 
      a beta of {round(posOLS.params[0],4)}, and a correlation of {round(pos_corr,4)}""")

senti_neg[0] = pd.to_numeric(senti_neg[0], errors='coerce')
senti_neg["RET"] = pd.to_numeric(senti_neg["RET"], errors='coerce')
negOLS = sm.OLS(senti_neg["RET"], sm.add_constant(senti_neg[0])).fit()
neg_corr = senti_neg[0].corr(senti_neg["RET"])
print(f"""The negative regression has an r-squared of {round(negOLS.rsquared,4)}, 
      a beta of {round(negOLS.params[0],4)}, and a correlation of {round(neg_corr,4)}""")

#Plots for sentiment analysis 
plt.scatter(senti_pos[0], senti_pos["RET"])
plt.title("Positive Scatter Plot")

plt.scatter(senti_neg[0], senti_neg["RET"], color = "red")
plt.title("Negative Scatter Plot")

# How has the language of financial performance evolved over time?
# We will answer this by comparing 2023 and 2024 10K and the most used noun chunks and adjectives
#Run functions on the documents again to reset string variables
files3 = []
for key in file_dict:
    pdf_path = file_dict[key]
    globals()[key] = pdf_reader2(pdf_path)
    globals()[key] = most_common_words(globals()[key])
    plot = plot_top_words(globals()[key], key)
    files3.append(globals()[key])
    
# When reading the 10Ks, there is clear consistency. Upon further review,
# the lanaguage used in these 10Ks have had little to no change over the subsequent years

