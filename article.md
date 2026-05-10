# Aeon for Time Series Clustering with Python Aeon is an open-source time series library for TS classification,
regression, clustering, forecasting, and transformation. It seems to be...

### Aeon for Time Series Clustering with Python
**Aeon** is an open-source time series library for TS classification, regression, clustering, forecasting, and transformation. It seems to be well maintained and there are articles about it from 2024.

> Warning --- I could only get Aeon to work with Pandas==1.4.0 (an older
> version). Newer versions of Pandas use a different index API which
> breaks a lot of time series libraries.

My favorite part are the visualizations which are pretty. But I miss the flexibility that I have with Matplotlib.

#### Let's take a look at visualizing Time Series with Aeon
Aeon includes plotting utilities for exploratory data analysis. Let's look at a basic Line Plot.


Not bad. Nothing spectactular but simple and nice.

### Time Series clustering and classification
Some time series follow predictable patterns but it can be hard to distinguish to which group those series belong. Given a random signal, how do we classify it with other similar signals?

This is where clustering algorithms help. The code generates 50 random samples that follow one of three set patterns (sine, cosine, or sine(2x)). Then we use aeon to classify which pattern each sample belongs to. It uses the k-nearest neighbor algo do sort things out. As a result, we can easily separate the different series into like buckets. From here, we could do more analysis on a specific bucket. Looking all all these series together just appears like noise. So the clustering really helps us in this case.


These graphs are much prettier than the line plot of the Passangers dataset.

We can take these labels and have aeon classify the data using a classifier. Because this is simulated data, i'm not surprised that the algo perfectly separates things.



Aeon is supposed to have an 'ETSForecaster' but I couldn't get this to work.

### So what?
Aeon classification, regression, clustering, and visualization work well. The API is simple and well documented. Compared to other tools, though, Aeon is not my favorite and I don't intent to use this one often.

#### Code
The code for this project is available on [GitHub](https://github.com/kylejones200/time_series/blob/main/medium/Aeon%20for%20Time%20Series%20Forecasting%20with%C2%A0Python.ipynb).

This is a different implementation using real data on [US Energy Generation.](https://www.eia.gov/electricity/data/browser/)
