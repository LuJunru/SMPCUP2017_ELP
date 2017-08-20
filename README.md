SMPCUP2017
Team：ELP
Organization：University of International Relations
Members：Lu Junru; Chen Le; Meng Kongming; Wang Fengyi; Xiangjun; Zhou Kaimin; Dong Zhenyuan; Shan jiawei; Lian Lingchen
-----------------------------------------------------------------------------
Introduction:
this repository is established to review codes and documents for our team ELP in SMPCUP2017, which is a user profiling contest
based on massive data provided by CSDN, including text, relations and interactions of users.
In SMPCUP2017, every team is requested to work on three specific tasks:
TASK1:Given a number of user documents (blogs or posts), generate 3 most appropriate keywords for each document.
      The generated keywords must appear in the document.
TASK2:Given the user's document information (blog or post) and behavior data (browsing, comment, collection, forwarding,
      point-of-thumb, step-by-step, private, etc.) for each user, mark the three most relevant interests for each user.
      The label space is given by CSDN.
TASK3:Given a number of users in a period of time (at least 1 year) document information (blog or post) and behavioral data
      (browsing, commentary, collection, forwarding, dating, attention, private messages, etc.),
      predict each user in the future over a period of time (half a year to 1 year) growth value.
      User growth is based on the user's overall performance scoring income, but will not publish the specific score criteria.
      The growth value will be normalized to the [0, 1] interval, where the value is 0 for user churn.
More detailed imformation could be browsed on the page: https://biendata.com/competition/smpcup2017/
-----------------------------------------------------------------------------
Baseline Models:
TASK1:TFIDF(Reference:https://en.wikipedia.org/wiki/Tf–idf#Term_frequency)
TASK2:W-BAG, a model based on Word2Vec and BaggingClassifer
TASK3:BPNN-XGboost(BPXG), a linear average model of BackPropagationNeuralNetwork and XGboost
-----------------------------------------------------------------------------
Final Models:
TASK1:S-TFIDF, a promoted model based on TFIDF and Textrank.
TASK2:S-TFIDF-LinerSVC-Word2Vec-Stacking(SLSS), a stacking model that using S-TFIDF as first layer and using a combination of
      LinearSVC and Word2Vec as second layer.
TASK3:PAR/GDR-NuSVR-Stacking(PGNS), a stacking model that using PassiveActiveRegressor and GrandientBoostingRegressor
      as first layer and using NuSVR as second layer.
-----------------------------------------------------------------------------
Performance:
TASK1  | TRAIN | VALID | TEST |
TFIDF  | 0.56  | 0.52  | None |
S-TFIDF| 0.61  | 0.56  | 0.56 |
---    | ---   | ---   | ---  |
TASK2  | TRAIN | VALID | TEST |
W-BAG  | None  | 0.40  | 0.373|
SLSS   | None  | 0.39  | 0.378|
---    | ---   | ---   | ---  |
TASK3  | TRAIN | VALID | TEST |
BPXG   | 0.54  | 0.59  | None |
PGNS   | 0.765 | 0.73  | 0.75 |
-----------------------------------------------------------------------------
Environment:
Task1: python 2.7
Task2: python 3.0
Task3: python 2.7
-----------------------------------------------------------------------------
More question:
lujunru31415926@163.com
