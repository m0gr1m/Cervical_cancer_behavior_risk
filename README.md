# Cervical cancer behavior risk

> <img src="https://i.imgur.com/DT54FHQ.png"  width="100" align="left">
>
>Cervical cancer is the second most common type of cancer that affects women around the world. 
Especially in developed countries. As with all cancers, early detection offers the best chance for successful treatment, 
so the ability to use behavioral science, which does not require expensive testing, can have a positive effect 
on initial diagnosis. 

[Link to presentation.](https://slideplayer.com/slide/11492208/) <br />
[Link to article.](https://www.ingentaconnect.com/content/asp/asl/2016/00000022/00000010/art00111;jsessionid=fmpbofs3ept7m.x-ic-live-02)

----

[Main Report.](https://github.com/m0gr1m/Cervical_cancer_behavior_risk/blob/main/reports/Cervical_Cancer_Report.pdf) <br />
[Report - Data overview.](https://github.com/m0gr1m/Cervical_cancer_behavior_risk/blob/main/reports/raport_data_overview.pdf) <br />
[Report - Model Results.](https://github.com/m0gr1m/Cervical_cancer_behavior_risk/blob/main/reports/raport_models_tuning.pdf)

----

## Variables 

The variables come from widely studied behavioral theory, more specifically:
+ The Health Belief Model (HBM)
+ Protection Motivation Theory (PMT)
+ Theory of Planned Behavior (TPB)
+ Social Cognitive Theory (SCT)

and consist of: 
+ Intention
+ Attitude
+ Subjective Norm
+ Perception
+ Motivation
+ Social Support
+ Empowerment

**Variable’s Indicator:** 

Prevention Behavior of Cervical Cancer: 
+ Y1 : Not put up sexual intercourse risk HPV infection 
+ Y2 : Consume nutritious food balanced 
+ Y3 : Personal hygiene 

Intention: 
+ Y4 : Aggregation 
+ Y5 : Compatibility 
+ Y6 : Commitment 

Attitude: 
+ Y7 : Direction to behavior prevention 
+ Y8 : Consistency 
+ Y9 : Spontaneity 

Subjective Norms: 
+ Y10 : Trust of norms 
+ Y11 : Significant Person 
+ Y12 : The fulfillment of norms which is believed to be 

Perception: 
+ Y13 : Susceptibility the perceived. 
+ Y14 : Potential severity the perceived 
+ Y15 : Perceived advantage 

Motivation: 
+ Y16 : The strength of a willingness to conduct prevention 
+ Y17 : The number of time provided to behavior prevention 
+ Y18 : Mutual consent leave other task by behavior prevention 

Social Support: 
+ X1 : Emotional given the other related behavior prevention 
+ X2 : Instrumental given the other related behavior prevention 
+ X3 : The information given the other related behavior prevention 

Empowerment: 
+ X4 : Provisions for needs in preventing 
+ X5 : Provisions for the ability to manage behavior prevention 
+ X6 : Provisions for the ability determines the way prevention 

----

## Dataset

This dataset consist of 18 variables that are provided by the seven indicators listed above and dependent variable 
which describes whether respondent has cervical cancer (_1=has cervical cancer, 0=no cervical cancer_): 
1) behavior_eating
2) behavior_personalHygiene
3) intention_aggregation
4) intention_commitment
5) attitude_consistency
6) attitude_spontaneity
7) norm_significantPerson
8) norm_fulfillment
9) perception_vulnerability
10) perception_severity
11) motivation_strength
12) motivation_willingness
13) socialSupport_emotionality
14) socialSupport_appreciation
15) socialSupport_instrumental
16) empowerment_knowledge
17) empowerment_abilities
18) empowerment_desires
19) **ca_cervix** 

Number of respondents: 72 <br />
Missing Data: 0

**Two classification algorithms were used in this study:**
+ Naive Bayes (NB)
+ Logistic Regression (LR)

A 10-fold cross validation was applied for each of them, and the results obtained are as follows:
+ NB: 
  + Accuracy: 91.67%
  + AUC: 0.96 
+ LR: 
  + Accuracy: 87.50%
  + AUC: 0.97

----

## Aim of the work

+ Improving the results obtained by researchers.
+ Presenting additional metrics that seem more appropriate for medical issues, e.g. precision and recall.

----

## Summary of results

Because the KMO test result showed a middling level of effectiveness we plotted a PCA graph for the two components that explain approximately 47% of the total variance of the dataset. As can be seen with the ellipse, it is possible to mark the area for cases with cancer fairly correctly. 

<img src="https://i.imgur.com/KXgcoCt.png"  width="700">

Using Python and Sklearn, the researchers' results were also improved. First, a group representing 25% of the size of the dataset was separated in a way that included, in appropriate numbers, cases from the minority class (*those with cancer*). In each case, the model was trained on 75% of the original dataset using 10-fold cross validation.

### Logistic Regression (LR)

+ data were standardized 
+ tuned parameters: class weight, C
+ model accuracy: 0.94667

ROC curve 

<img src="https://i.imgur.com/tncvMlT.png"  width="600">

Since we were dealing with an unbalanced number among the classes (*less 1/3 of the cases have cancer*) G-Mean was used to determine the best cut-off point and with it, a test of the model's effectiveness was performed on test data (*25% that the model had never seen before*).

**Results after moving the cut-off point.**

<img src="https://i.imgur.com/yAaJN1S.png"  width="400">

### Naive Bayes (NB)

+ data have not been standardized 
+ tuned parameters: alpha 
+ model accuracy: 0.87

The same procedure as for logistic regression was used.

ROC curve 

<img src="https://i.imgur.com/LuXTnHc.png"  width="600">

**Results after moving the cut-off point.**

<img src="https://i.imgur.com/VsTvrCb.png"  width="400">

## Conclusion

As can be observed, the results we obtained for Naive Bayes are excellent. We achieved 100% correctness in every metric, which means that the model flawlessly detects cancer cases based on behavioral studies. 
