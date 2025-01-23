# Unsupervised Learning Clustering on Reasearch Papers from University of Warsaw Faculty of Economic Sciences

The general goal of this research was to find out which topics are the area of research by each member of the University of Warsaw Faculty of Economic Sciences academia.


This code aims to provide a scraper in order to conduct semantic clustering on research papers from available supervisors on UW's Faculty of Economic Sciences.

## The Scraper

The scraper is crawling through the Google Scholar pages for provided authors in the list.txt file gathering all of the papers' details.
I am using Selenium, because of the easy session and multiple-tab handling. 

## Clustering Methodology

### Preprocessing

To conduct clustering on text data first the data needed preprocessing in order to get a proper standardized dataset that can be used in clustering.
The following steps were taken:
1) Translation of all texts into English
2) Tokenizing the texts
3) Removing the stopwords
4) Lemmatizing all tokens that are left

### TF - IDF Vectorizing

I used a ready implementation of TfidfVectorizer from sklearn in order to calculate the TF - IDF for all of the papers and additionally prepare a embedded vector representation of the text for more accurate similarity search.
It returns the document-term matrix, which is ready to be used in further operations.

### Dimensionality Reduction with UMAP

In order to properly cluster our data we need to reduce the number of features in the vectorized data. High dimensional text data is not really well suitable for clustering.
To enchance it we reduce the number of manifolds, so that it is more managable by DBSCAN.


### Clustering proper - HDBSCAN

The algorithm I decided to use was DBSCAN, with some reading I found out that HDBSCAN ( Hierarchical DBSCAN ) is a much better choice as it seems to handle the noise even smoother, which I expected to be high in text data.
One other advantage is that the radius is not necessary to be set as it is determined based on hierarchical distances. 


## Results

![clustering TSNE](https://github.com/user-attachments/assets/5d5678fb-107d-4c7d-b827-4a2c72bbda08)


### Cluster Top Terms and Coherences

Top Terms per Cluster (sorted by coherence scores):
Cluster 10: wealth, household, income, distribution, net, project, difference, inequality, country, consumption
Cluster 5: constitutional, political, rule, society, country, gap, country period, panel, economic, period
Cluster 4: trade, gravity, country, variable, eu, eurozone, cee, cee country, agreement, human
Cluster 14: spatial, location, model, different, learning, distance, point, method, topic, firm
Cluster 17: pension, age, population, defined, welfare, contribution, people, increasing, benefit, life
Cluster 1: firm, innovation, export, country, productivity, foreign, performance, exporting, empirical, activity
Cluster 15: cloud, model, ii, simulation, effect, evolution, performed, numerical, dimension, cycle
Cluster 16: preference, choice, support, respondent, service, valuation, environmental, survey, increase, experiment
Cluster 12: public, local, government, sector, spending, limit, expenditure, good, revenue, city
Cluster 0: monetary, rate, migration, policy, exchange, exchange rate, monetary policy, inflation, euro, economy
Cluster 2: growth, cycle, business, expenditure, concentration, effect, country, period, sectoral, result
Cluster 8: school, student, education, economics, result, reform, year, test, international, poland
Cluster 11: tax, income, household, redistributive, benefit, inequality, debt, country, fiscal, social
Cluster 3: regional, wage, region, employment, poland, labor, market, worker, labour, policy
Cluster 7: gas, emission, engine, method, cost, energy, curve, equilibrium, sector, general equilibrium
Cluster 13: trust, article, research, factor, technology, network, determinant, exposure, value, aim
Cluster 9: bank, financial, crisis, central bank, reporting, central, banking, economic, information, law
Cluster 6: volatility, strategy, market, investment, model, forecast, risk, portfolio, price, stock
Cluster -1: health, market, economy, factor, policy, poland, care, change, private, learning

Coherence Scores (sorted):
Cluster 10: 0.1440
Cluster 5: 0.1345
Cluster 4: 0.1137
Cluster 14: 0.1084
Cluster 17: 0.1071
Cluster 1: 0.1058
Cluster 15: 0.1010
Cluster 16: 0.0993
Cluster 12: 0.0958
Cluster 0: 0.0944
Cluster 2: 0.0898
Cluster 8: 0.0874
Cluster 11: 0.0871
Cluster 3: 0.0830
Cluster 7: 0.0797
Cluster 13: 0.0757
Cluster 9: 0.0693
Cluster 6: 0.0682
Cluster -1: 0.0451

### Authors and Clusters
Clusters and Paper Counts for the Authors:
Katarzyna Kopczewska:
  Noise: 1 papers
  Cluster 1: 1 papers
  Cluster 2: 1 papers
  Cluster 3: 3 papers
  Cluster 6: 1 papers
  Cluster 11: 6 papers
  Cluster 12: 1 papers
  Cluster 14: 11 papers
Agata Kocia:
  Noise: 2 papers
  Cluster 9: 4 papers
  Cluster 11: 10 papers
  Cluster 12: 1 papers
  Cluster 13: 3 papers
Agnieszka Kopańska:
  Noise: 1 papers
  Cluster 12: 6 papers
Andrzej Cieślik:
  Noise: 2 papers
  Cluster 1: 19 papers
  Cluster 2: 4 papers
  Cluster 3: 8 papers
  Cluster 4: 18 papers
  Cluster 11: 2 papers
Łukasz Goczek:
  Noise: 1 papers
  Cluster 0: 4 papers
  Cluster 1: 1 papers
  Cluster 2: 4 papers
  Cluster 7: 1 papers
  Cluster 8: 1 papers
  Cluster 9: 1 papers
  Cluster 13: 3 papers
Jan Hagemejer:
  Cluster 1: 5 papers
  Cluster 2: 4 papers
  Cluster 4: 5 papers
  Cluster 17: 7 papers
Jan Michałek:
  Cluster 1: 4 papers
Jerzy Mycielski:
  Cluster 0: 1 papers
  Cluster 1: 5 papers
  Cluster 3: 2 papers
  Cluster 4: 11 papers
  Cluster 7: 1 papers
  Cluster 11: 1 papers
  Cluster 13: 1 papers
  Cluster 16: 1 papers
  Cluster 17: 1 papers
Anna Janicka:
  Cluster 3: 1 papers
  Cluster 4: 1 papers
  Cluster 7: 8 papers
  Cluster 9: 1 papers
  Cluster 13: 1 papers
  Cluster 15: 1 papers
Anna Nicińska:
  Noise: 4 papers
  Cluster 8: 1 papers
  Cluster 10: 1 papers
  Cluster 11: 1 papers
  Cluster 13: 2 papers
  Cluster 16: 4 papers
  Cluster 17: 3 papers
Grzegorz Kula:
  Noise: 3 papers
  Cluster 2: 2 papers
  Cluster 11: 2 papers
  Cluster 12: 3 papers
  Cluster 13: 1 papers
  Cluster 17: 3 papers
Bartłomiej Rokicki:
  Cluster 3: 6 papers
  Cluster 11: 1 papers
Ewa Aksman:
  Noise: 2 papers
  Cluster 7: 1 papers
  Cluster 11: 13 papers
  Cluster 12: 1 papers
Ewa Weychert:
  Noise: 1 papers
  Cluster 11: 1 papers
Grzegorz Wesołowski:
  Cluster 0: 11 papers
  Cluster 2: 1 papers
  Cluster 9: 4 papers
Juliusz Jabłecki:
  Cluster 0: 2 papers
  Cluster 6: 5 papers
  Cluster 9: 6 papers
Jacek Lewkowicz:
  Noise: 2 papers
  Cluster 1: 5 papers
  Cluster 4: 2 papers
  Cluster 5: 8 papers
  Cluster 9: 4 papers
  Cluster 15: 1 papers
Katarzyna Metelska-Szaniawska:
  Noise: 3 papers
  Cluster 3: 1 papers
  Cluster 5: 7 papers
  Cluster 9: 4 papers
  Cluster 11: 1 papers
Krzysztof Szczygielski:
  Noise: 2 papers
  Cluster 1: 12 papers
  Cluster 3: 2 papers
  Cluster 4: 1 papers
  Cluster 16: 1 papers
Tomasz Kopczewski:
  Noise: 3 papers
  Cluster 6: 1 papers
  Cluster 7: 1 papers
  Cluster 8: 5 papers
  Cluster 9: 3 papers
  Cluster 11: 1 papers
  Cluster 14: 1 papers
  Cluster 15: 1 papers
  Cluster 16: 1 papers
Jakub Michańków:
  Cluster 6: 9 papers
  Cluster 8: 1 papers
  Cluster 12: 1 papers
Paweł Sakowski:
  Cluster 4: 1 papers
  Cluster 6: 25 papers
  Cluster 8: 1 papers
  Cluster 12: 1 papers
Robert Ślepaczuk:
  Cluster 6: 36 papers
Marcin Bielecki:
  Cluster 0: 8 papers
  Cluster 2: 2 papers
  Cluster 7: 1 papers
  Cluster 9: 1 papers
  Cluster 11: 1 papers
  Cluster 17: 9 papers
Michał Brzozowski:
  Noise: 1 papers
  Cluster 0: 4 papers
  Cluster 2: 3 papers
  Cluster 4: 1 papers
  Cluster 6: 3 papers
  Cluster 11: 1 papers
  Cluster 12: 1 papers
  Cluster 13: 1 papers
Maria Kubara:
  Noise: 1 papers
  Cluster 1: 1 papers
  Cluster 3: 1 papers
  Cluster 14: 3 papers
Wojciech Grabowski:
  Cluster 1: 4 papers
  Cluster 15: 1 papers
Maciej Jakubowski:
  Cluster 1: 1 papers
  Cluster 3: 1 papers
  Cluster 4: 1 papers
  Cluster 6: 2 papers
  Cluster 8: 11 papers
  Cluster 11: 2 papers
Marcin Chlebus:
  Noise: 3 papers
  Cluster 2: 2 papers
  Cluster 6: 9 papers
  Cluster 13: 3 papers
  Cluster 17: 2 papers
Marcin Gruszczyński:
  Noise: 2 papers
  Cluster 0: 7 papers
  Cluster 9: 7 papers
  Cluster 11: 1 papers
  Cluster 13: 1 papers
Mehmet Burak Turgut:
  Cluster 3: 1 papers
  Cluster 4: 1 papers
  Cluster 11: 2 papers
Mikołaj Czajkowski:
  Noise: 1 papers
  Cluster 6: 1 papers
  Cluster 9: 1 papers
  Cluster 11: 3 papers
  Cluster 12: 1 papers
  Cluster 16: 8 papers
Olga Kiuila:
  Noise: 1 papers
  Cluster 6: 1 papers
  Cluster 7: 13 papers
  Cluster 11: 3 papers
Paweł Kaczmarczyk:
  Cluster 0: 7 papers
  Cluster 3: 4 papers
  Cluster 13: 2 papers
Piotr Boguszewski:
  Noise: 1 papers
  Cluster 0: 2 papers
  Cluster 1: 1 papers
  Cluster 8: 1 papers
  Cluster 9: 10 papers
Piotr Żoch:
  Cluster 0: 1 papers
Stanisław Cichocki:
  Noise: 1 papers
  Cluster 3: 2 papers
  Cluster 7: 1 papers
  Cluster 9: 2 papers
  Cluster 11: 1 papers
  Cluster 13: 2 papers

