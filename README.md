# Using Deep Learning to Hedge Rainbow Options

A thesis sent in partial fulfilment of the MSc 203 - Financial Markets of Paris Dauphine PSL University.

Written by Thibault Collin under the supervision of Prof. Thibaud Vienne and Prof. Gaëlle Le Fol during the 2022 spring semester.

The general scope of this thesis will be to further study the application of artificial neural networks in the context of hedging rainbow options. Due to their inherently complex features, such as the correlated paths that the prices of their underlying assets take or their absence from traded markets, finding an optimal hedging strategy for rainbow options is difficult, and traders usually have to resort to models and methods they know are inaccurate. An alternative approach involving deep learning however recently surfaced in the context of hedging vanilla options \cite{bu18}, and researchers have started to see potential in the use of neural networks for options endowed with more exotic features

The key to a near-perfect hedge for contingent claims might be hidden behind the training of neural network algorithms, and the scope of this research will be to further investigate how those innovative hedging techniques can be extended to rainbow options, using recent research, and to compare our results with those proposed by the current models and techniques used by traders, such as running \textit{Monte-Carlo} path simulations. In order to accomplish that, we will try to develop an algorithm capable of designing an innovative and optimal hedging strategy for rainbow options using some intuition developed to hedge vanilla options and price exotics. But although it was shown from past literature to be potentially efficient and cost-effective, the opaque nature of an artificial neural network will make it difficult for the deep learning algorithm to be fully trusted and used as a sole method for hedging purposes, but rather as an additional technique associated with other models.

(the algorithm was partly inspired by https://github.com/YuMan-Tam/deep-hedging)
