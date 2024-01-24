# Cryptocurrency Market Forecasting & Sentiment Analysis📈

<img src="https://github.com/bryan-ortiz0/ETHtractor/assets/130245932/c5db90ff-223e-441d-9f52-ceef27f7e29d" width="1000">

## Introduction
In this repository, we investigate cyrptocurrency market dynamics, focusing primarily on Ethereum (ETH)-USD pairs sourced from [Bitstamp](https://www.bitstamp.net/markets/eth/usd/) and incorporating sentiments derived from [Wikipedia revisions](https://en.wikipedia.org/w/index.php?title=Ethereum&action=history) by utilizing [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) from Hugging Face Transformers. Combining statistical tools, feature engineering, and machine learning models, we thoroughly examine trends, correlations, volatilities, and seasonalities apparent in closing prices, volumes, and sentiments. More importantly, we train and rigorously evaluate five distinct predictive models--**ARIMA**, **Random Forest**, **XGBoost**, **CNN**, and **LSTM**--aiming to pinpoint the most effective approach for generating accurate forecasts. Two critical metrics guide our evaluation: **R²** and **Mean Absolute Error (MAE)**.

Throughout the analysis, we focus on R² and MAE as they reveal essential facets of the predictive capacity of each model.

R², interpreted as the coefficient of determination, represents the ratio of variation in the dependent variable (closely tied to the goodness-of-fit measure) explained by the independent variable(s). Simply put, it gauges the percentage of deviation in the target attribute that the fitted model successfully explains. *Higher* R² values imply closer approximations to reality, reflecting better model fitness. Nevertheless, caution must be exercised when comparing models soley based on R² as this metric doesn't necessarily penalize models with excessive complexity. 

On the other hand, MAE offers a complementary perspective by measuring the average magnitude of absolute differences between predicted and actual values. *Lower* MAEs denote more precise predictions, revealing reduced dispersion surrounding the target. While resistant to extreme values, MAE fails to distinguish between positive and negative deviations, thus rendering it insensitive to systematic biases.

Using both R² and MAE together ensures balanced scrutiny of moel performance, accounting for goodness-of-fit and error magnitudes. We strive to select the most proficient model capable of producing lucid, reliable forecasts.

## Exploratory Data Analysis
First, let us inspect the Ethereum hourly price data retrieved from Bitstamp, covering the period from May 2018 till January 2024, as visualized below. The green and red price bars overlap illustrates the high volatility characteristic of cryptocurrency markets.

![eth_tableau](https://github.com/bryan-ortiz0/ETHtractor/assets/130245932/7ce93a62-f283-45af-b31e-4dad2a765dfd)
*Made in Tableau*

## Preliminary Visual Inspection
As evident in the plot, dramatic price swings dominate the scene, punctuated by occasional sharp peaks and troughs, characteristic of cryptocurrency markets. 

Zooming onto 2021 reveals striking price action for Ethereum.

![ethereum_closing_prices_2021_daily_weekly_monthly_views](https://github.com/bryan-ortiz0/ETHtractor/assets/130245932/76fdd41e-2e0e-46eb-9ae3-384cb3cb2725)

## Sentiment Analysis
I included sentiment anaysis derived from Wikipedia revisions associated with Ethereum. Interestingly enough, negative sentiment towards Ethereum was at its highest while price was also surging to new all time highs but sentiment is shifting as the technology matures.

![ethereum_sentiment_and_neg_sentiment](https://github.com/bryan-ortiz0/ETHtractor/assets/130245932/b1989a65-77c3-4ce0-b2e6-8dd41d728245)

## Stationarity Checks
Next, we look at the autocorrelation within our dataset with lots of noise and no obvious lags that stand out as statistically significant.

![ethereum_autcorrelation](https://github.com/bryan-ortiz0/ETHtractor/assets/130245932/f2260f27-45e4-49df-949e-b0b5026b223f)

| Adfuller | ADF Statistic | p-Value | 
| :------: | :------------:| :-----: |
| close    | -1.50         | 0.53    |
| close_log| -0.79         | 0.82    |

ADF statistic and p-value indicate weak evidence against the null hypothesis, suggesting nonstationarity.

## Model Training & Evaluation
### 1. AutoRegressive Integrated Moving Average (ARIMA (4,1,0))
Classical linear model for stationary time-series forecasting, composed of autoregressive (AR(4)), integrated (I(1)), and moving average (MA(0)) components. Applicable for short-term forecasting, assuming no significant structural variations exist in the series. Serves as a baseline reference point to gauge effectiveness of other models.
### 2. Random Forest (RF)
Ensemble method integrating multiple decision tree learners, mitigating risk of overfitting. Utilizes bootstrap sampling, feature randomness, and aggregating outputs for heightened precision and stability. Effectively manages non-stationarity, non-linearity, and noise in vast feature domains.
### 3. eXtreme Gradient Boosting (XGBoost)
Potent gradient-boosting mechanism augmenting traditional Generalized Linear Models (GLMs) with native support for managing missing values. Delivers exceptional efficacy, versatility, and regulatory controls. Adapable to complex patterns, captitalizing on nuanced interactions amongst features.
### 4. Convolutional Neural Network (CNN)
Deep learning design originally developed for image recognition tasks, repurposed for sequence prediction challenges. Comprises convolution filters, pooling layers, and fully connected networks to construct hierarchical abstractions. Efficient at discerning regional motifs, periodicities, and abrupt transitions common in cryptocurrency markets.
### 5. Long Short-Term Memory (LSTM)
Specialized Recurrent Neural Network (RNN) engineered for handling sequential data mining tasks. Features sophisticated gatekeeping mechanisms, maintaining context awareness across evolving situations. Mitigates vanishing gradients, preserving informational continuity amidst dynamic market scenarios, frequently encountered in cryptocurrencies.

## Model Metrics
| Model | R² | MAE |
| :---: | :-: | :-: |
| ARIMA (4,1,0) | TBD | TBD|
| Random Forest | TBD | TBD |
| XGBoost | TBD | TBD |
| CNN | TBD | TBD | 
| LSTM | TBD | TBD |

## Optimal Model for Holdout Data
Finally, having trained and evaluated the models, we choose the top-performing model, 'model_name', and validate on the holdout data set. This step aims to validate the model's ability to produce accurate forecasts beyond the initial training and testing intervals.

## Summary
We explored the exciting of cryptocurrency analytics, combining sentiment analysis and machine learning techniques to illuminate hidden patterns and derive meaningful insights. Through diligent preparation, meticulous model construction, and rigorous evaluation, we identified potent contenders for profitable investment guidance. 

## Future Work
- [ ] Incorporating social media feeds and forum discussions to supplement sentiment analysis.
- [ ] Experimenting with transfer learning and domain adaptation to leverage pretrained models.
- [ ] Expand analysis with multiple datasets.
