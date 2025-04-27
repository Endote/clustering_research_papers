# Unsupervised Learning Clustering and Dimension Reduction on Reasearch Papers from University of Warsaw Faculty of Economic Sciences

The general goal of this research was to find out which topics are the area of research by each member of the University of Warsaw Faculty of Economic Sciences academia.


This code aims to provide a scraper in order to conduct semantic clustering on research paper abstracts from available supervisors on UW's Faculty of Economic Sciences.

# The Scraper

The scraper is crawling through the Google Scholar pages for provided authors in the list.txt file gathering all of the papers' details.
I am using Selenium, because of the easy session and multiple-tab handling. 

# Analysis Methodology

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


### Clustering with HDBSCAN

The algorithm I decided to use was DBSCAN, with some reading I found out that HDBSCAN ( Hierarchical DBSCAN ) is a much better choice as it seems to handle the noise even smoother, which I expected to be high in text data.
One other advantage is that the radius is not necessary to be set as it is determined based on hierarchical distances. 


# Results

![clustering TSNE](https://github.com/user-attachments/assets/5d5678fb-107d-4c7d-b827-4a2c72bbda08)


### Cluster Top Terms and Coherences

Top Terms per Cluster (sorted by coherence scores):<br/>
Cluster 10: wealth, household, income, distribution, net, project, difference, inequality, country, consumption <br/>
Cluster 5: constitutional, political, rule, society, country, gap, country period, panel, economic, period<br/>
Cluster 4: trade, gravity, country, variable, eu, eurozone, cee, cee country, agreement, human<br/>
Cluster 14: spatial, location, model, different, learning, distance, point, method, topic, firm<br/>
Cluster 17: pension, age, population, defined, welfare, contribution, people, increasing, benefit, life<br/>
Cluster 1: firm, innovation, export, country, productivity, foreign, performance, exporting, empirical, activity<br/>
Cluster 15: cloud, model, ii, simulation, effect, evolution, performed, numerical, dimension, cycle<br/>
Cluster 16: preference, choice, support, respondent, service, valuation, environmental, survey, increase, experiment<br/>
Cluster 12: public, local, government, sector, spending, limit, expenditure, good, revenue, city<br/>
Cluster 0: monetary, rate, migration, policy, exchange, exchange rate, monetary policy, inflation, euro, economy<br/>
Cluster 2: growth, cycle, business, expenditure, concentration, effect, country, period, sectoral, result<br/>
Cluster 8: school, student, education, economics, result, reform, year, test, international, poland<br/>
Cluster 11: tax, income, household, redistributive, benefit, inequality, debt, country, fiscal, social<br/>
Cluster 3: regional, wage, region, employment, poland, labor, market, worker, labour, policy<br/>
Cluster 7: gas, emission, engine, method, cost, energy, curve, equilibrium, sector, general equilibrium<br/>
Cluster 13: trust, article, research, factor, technology, network, determinant, exposure, value, aim<br/>
Cluster 9: bank, financial, crisis, central bank, reporting, central, banking, economic, information, law<br/>
Cluster 6: volatility, strategy, market, investment, model, forecast, risk, portfolio, price, stock<br/>
Cluster -1: health, market, economy, factor, policy, poland, care, change, private, learning<br/>
<br/><br/>
Coherence Scores (sorted):<br/>
Cluster 10: 0.1440<br/>
Cluster 5: 0.1345<br/>
Cluster 4: 0.1137<br/>
Cluster 14: 0.1084<br/>
Cluster 17: 0.1071<br/>
Cluster 1: 0.1058<br/>
Cluster 15: 0.1010<br/>
Cluster 16: 0.0993<br/>
Cluster 12: 0.0958<br/>
Cluster 0: 0.0944<br/>
Cluster 2: 0.0898<br/>
Cluster 8: 0.0874<br/>
Cluster 11: 0.0871<br/>
Cluster 3: 0.0830<br/>
Cluster 7: 0.0797<br/>
Cluster 13: 0.0757<br/>
Cluster 9: 0.0693<br/>
Cluster 6: 0.0682<br/>
Cluster -1: 0.0451<br/>

### Authors and Clusters
Clusters and Paper Counts for the Authors:<br/>
Katarzyna Kopczewska:<br/>
  Noise: 1 papers<br/>
  Cluster 1: 1 papers<br/>
  Cluster 2: 1 papers<br/>
  Cluster 3: 3 papers<br/>
  Cluster 6: 1 papers<br/>
  Cluster 11: 6 papers<br/>
  Cluster 12: 1 papers<br/>
  Cluster 14: 11 papers<br/>
  <br/>
Agata Kocia:<br/>
  Noise: 2 papers<br/>
  Cluster 9: 4 papers<br/>
  Cluster 11: 10 papers<br/>
  Cluster 12: 1 papers<br/>
  Cluster 13: 3 papers<br/>
<br/>
Agnieszka Kopańska:<br/>
  Noise: 1 papers<br/>
  Cluster 12: 6 papers<br/>
<br/>
Andrzej Cieślik:<br/>
  Noise: 2 papers<br/>
  Cluster 1: 19 papers<br/>
  Cluster 2: 4 papers<br/>
  Cluster 3: 8 papers<br/>
  Cluster 4: 18 papers<br/>
  Cluster 11: 2 papers<br/>
<br/>
Łukasz Goczek:<br/>
  Noise: 1 papers<br/>
  Cluster 0: 4 papers<br/>
  Cluster 1: 1 papers<br/>
  Cluster 2: 4 papers<br/>
  Cluster 7: 1 papers<br/>
  Cluster 8: 1 papers<br/>
  Cluster 9: 1 papers<br/>
  Cluster 13: 3 papers<br/>
<br/>
Jan Hagemejer:<br/>
  Cluster 1: 5 papers<br/>
  Cluster 2: 4 papers<br/>
  Cluster 4: 5 papers<br/>
  Cluster 17: 7 papers<br/>
<br/>
Jan Michałek:<br/>
  Cluster 1: 4 papers<br/>
<br/>
Jerzy Mycielski:<br/>
  Cluster 0: 1 papers<br/>
  Cluster 1: 5 papers<br/>
  Cluster 3: 2 papers<br/>
  Cluster 4: 11 papers<br/>
  Cluster 7: 1 papers<br/>
  Cluster 11: 1 papers<br/>
  Cluster 13: 1 papers<br/>
  Cluster 16: 1 papers<br/>
  Cluster 17: 1 papers<br/>
<br/>
Anna Janicka:<br/>
  Cluster 3: 1 papers<br/>
  Cluster 4: 1 papers<br/>
  Cluster 7: 8 papers<br/>
  Cluster 9: 1 papers<br/>
  Cluster 13: 1 papers<br/>
  Cluster 15: 1 papers<br/>
<br/>
Anna Nicińska:<br/>
  Noise: 4 papers<br/>
  Cluster 8: 1 papers<br/>
  Cluster 10: 1 papers<br/>
  Cluster 11: 1 papers<br/>
  Cluster 13: 2 papers<br/>
  Cluster 16: 4 papers<br/>
  Cluster 17: 3 papers<br/>
<br/>
Grzegorz Kula:<br/>
  Noise: 3 papers<br/>
  Cluster 2: 2 papers<br/>
  Cluster 11: 2 papers<br/>
  Cluster 12: 3 papers<br/>
  Cluster 13: 1 papers<br/>
  Cluster 17: 3 papers<br/>
<br/>
Bartłomiej Rokicki:<br/>
  Cluster 3: 6 papers<br/>
  Cluster 11: 1 papers<br/>
<br/>
Ewa Aksman:<br/>
  Noise: 2 papers<br/>
  Cluster 7: 1 papers<br/>
  Cluster 11: 13 papers<br/>
  Cluster 12: 1 papers<br/>
<br/>
Ewa Weychert:<br/>
  Noise: 1 papers<br/>
  Cluster 11: 1 papers<br/>
<br/>
Grzegorz Wesołowski:<br/>
  Cluster 0: 11 papers<br/>
  Cluster 2: 1 papers<br/>
  Cluster 9: 4 papers<br/>
<br/>
Juliusz Jabłecki:<br/>
  Cluster 0: 2 papers<br/>
  Cluster 6: 5 papers<br/>
  Cluster 9: 6 papers<br/>
<br/>
Jacek Lewkowicz:<br/>
  Noise: 2 papers<br/>
  Cluster 1: 5 papers<br/>
  Cluster 4: 2 papers<br/>
  Cluster 5: 8 papers<br/>
  Cluster 9: 4 papers<br/>
  Cluster 15: 1 papers<br/>
<br/>
Katarzyna Metelska-Szaniawska:<br/>
  Noise: 3 papers<br/>
  Cluster 3: 1 papers<br/>
  Cluster 5: 7 papers<br/>
  Cluster 9: 4 papers<br/>
  Cluster 11: 1 papers<br/>
<br/>
Krzysztof Szczygielski:<br/>
  Noise: 2 papers<br/>
  Cluster 1: 12 papers<br/>
  Cluster 3: 2 papers<br/>
  Cluster 4: 1 papers<br/>
  Cluster 16: 1 papers<br/>
<br/>
Tomasz Kopczewski:<br/>
  Noise: 3 papers<br/>
  Cluster 6: 1 papers<br/>
  Cluster 7: 1 papers<br/>
  Cluster 8: 5 papers<br/>
  Cluster 9: 3 papers<br/>
  Cluster 11: 1 papers<br/>
  Cluster 14: 1 papers<br/>
  Cluster 15: 1 papers<br/>
  Cluster 16: 1 papers<br/>
<br/>
Jakub Michańków:<br/>
  Cluster 6: 9 papers<br/>
  Cluster 8: 1 papers<br/>
  Cluster 12: 1 papers<br/>
<br/>
Paweł Sakowski:<br/>
  Cluster 4: 1 papers<br/>
  Cluster 6: 25 papers<br/>
  Cluster 8: 1 papers<br/>
  Cluster 12: 1 papers<br/>
<br/>
Robert Ślepaczuk:<br/>
  Cluster 6: 36 papers<br/>
<br/>
Marcin Bielecki:<br/>
  Cluster 0: 8 papers<br/>
  Cluster 2: 2 papers<br/>
  Cluster 7: 1 papers<br/>
  Cluster 9: 1 papers<br/>
  Cluster 11: 1 papers<br/>
  Cluster 17: 9 papers<br/>
<br/>
Michał Brzozowski:<br/>
  Noise: 1 papers<br/>
  Cluster 0: 4 papers<br/>
  Cluster 2: 3 papers<br/>
  Cluster 4: 1 papers<br/>
  Cluster 6: 3 papers<br/>
  Cluster 11: 1 papers<br/>
  Cluster 12: 1 papers<br/>
  Cluster 13: 1 papers<br/>
<br/>
Maria Kubara:<br/>
  Noise: 1 papers<br/>
  Cluster 1: 1 papers<br/>
  Cluster 3: 1 papers<br/>
  Cluster 14: 3 papers<br/>
<br/>
Wojciech Grabowski:<br/>
  Cluster 1: 4 papers<br/>
  Cluster 15: 1 papers<br/>
<br/>
Maciej Jakubowski:<br/>
  Cluster 1: 1 papers<br/>
  Cluster 3: 1 papers<br/>
  Cluster 4: 1 papers<br/>
  Cluster 6: 2 papers<br/>
  Cluster 8: 11 papers<br/>
  Cluster 11: 2 papers<br/>
<br/>
Marcin Chlebus:<br/>
  Noise: 3 papers<br/>
  Cluster 2: 2 papers<br/>
  Cluster 6: 9 papers<br/>
  Cluster 13: 3 papers<br/>
  Cluster 17: 2 papers<br/>
<br/>
Marcin Gruszczyński:<br/>
  Noise: 2 papers<br/>
  Cluster 0: 7 papers<br/>
  Cluster 9: 7 papers<br/>
  Cluster 11: 1 papers<br/>
  Cluster 13: 1 papers<br/>
<br/>
Mehmet Burak Turgut:<br/>
  Cluster 3: 1 papers<br/>
  Cluster 4: 1 papers<br/>
  Cluster 11: 2 papers<br/>
<br/>
Mikołaj Czajkowski:<br/>
  Noise: 1 papers<br/>
  Cluster 6: 1 papers<br/>
  Cluster 9: 1 papers<br/>
  Cluster 11: 3 papers<br/>
  Cluster 12: 1 papers<br/>
  Cluster 16: 8 papers<br/>
<br/>
Olga Kiuila:<br/>
  Noise: 1 papers<br/>
  Cluster 6: 1 papers<br/>
  Cluster 7: 13 papers<br/>
  Cluster 11: 3 papers<br/>
<br/>
Paweł Kaczmarczyk:<br/>
  Cluster 0: 7 papers<br/>
  Cluster 3: 4 papers<br/>
  Cluster 13: 2 papers<br/>
<br/>
Piotr Boguszewski:<br/>
  Noise: 1 papers<br/>
  Cluster 0: 2 papers<br/>
  Cluster 1: 1 papers<br/>
  Cluster 8: 1 papers<br/>
  Cluster 9: 10 papers<br/>
<br/>
Piotr Żoch:<br/>
  Cluster 0: 1 papers<br/>
<br/>
Stanisław Cichocki:<br/>
  Noise: 1 papers<br/>
  Cluster 3: 2 papers<br/>
  Cluster 7: 1 papers<br/>
  Cluster 9: 2 papers<br/>
  Cluster 11: 1 papers<br/>
  Cluster 13: 2 papers<br/>

# Conclusions
This report detailed the process and results of an unsupervised learning clustering analysis conducted on research paper abstracts from faculty members of the University of Warsaw's Faculty of Economic Sciences.  By employing web scraping, text preprocessing, TF-IDF vectorization, dimensionality reduction with UMAP, and HDBSCAN clustering, we successfully identified distinct research topics within the members of the Academia on the Faculty of Economics in University of Warsaw.<br/>

There is some solid proof of success for this clustering, for example the papers of<br/> 
https://www.wne.uw.edu.pl/members/profile/view/41 <br/>
Robert Ślepaczuk:<br/>
  Cluster 6: 36 papers<br/>

  Cluster 6: volatility, strategy, market, investment, model, forecast, risk, portfolio, price, stock<br/>

These results give us a general insight into what topics are interesting for which members of the academia. This could help potential students to pursue their desired scientific goal under proper supervisory of an adequate teacher.

Further research could explore alternative clustering algorithms or preprocessing techniques to refine the results and potentially uncover more nuanced thematic distinctions.

### Appendix
List of authors whose research papers were used in this text-topic analysis:<br/> 
https://scholar.google.com/citations?hl=pl&user=XSzsEtwAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=i7awptIAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=Mg7Z2qoAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=q9zFJHkAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=tRjklFQAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=c_OKnLEAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=PhNAgr0AAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=ZGV1xWQAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=iTq62wYAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=C8aGtDsAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=J9nzkEUAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=8-4v8SMAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=Yah04gwAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=N-ihQMoAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=vwmdy4EAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=hXO5VfYAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=Vg4noNIAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=UCuM1lwAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=zIMX4HEAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=mIcMty8AAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=uzVDnVsAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=vgzkd9IAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=SzMb1DUAAAAJ <br/>  
https://scholar.google.com/citations?hl=pl&user=sj_axrYAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=LBovu9gAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=rQh_QU0AAAAJ <br/>  
https://scholar.google.com/citations?hl=pl&user=VMDN_VgAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=s9ocZnkAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=9hDUxqwAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=evkeykQAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=L2Y9PJcAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=BgbTaQkAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=SBxcIDgAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=WVjmW3wAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=_XLs52sAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=iJGLqokAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=dwY3t_MAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=gD5nAgQAAAAJ <br/> 
https://scholar.google.com/citations?hl=pl&user=vqCpol8AAAAJ <br/> 
 


















