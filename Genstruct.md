March 2024 (?) - [[Nous Research]]
HuggingFace Card: [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B)

Inspired by earlier work from Shanghai: [[Ada-Instruct]]

Summary
> Genstruct ==7B== is an ==instruction-generation model==, designed to ==create valid instructions given a raw text corpus==. This ==enables the creation of new, partially synthetic instruction finetuning datasets from any raw-text corpus.==
> Previous methods largely rely on in-context approaches to generate instructions, while Ada-Instruct trained a custom instruction-generation model.
> Inspired by this, we took this approach further by grounding the generations in user-provided context passages. Further, the model is trained to generate questions involving complex scenarios that require detailed reasoning, allowing for models trained on the generated data to reason step-by-step.

Examples

Law of Large Numbers
```
[[[Title]]] Law of large numbers
[[[Content]]] In probability theory, the law of large numbers (LLN) is a mathematical theorem that states that the average of the results obtained from a large number of independent and identical random samples converges to the true value, if it exists.[1] More formally, the LLN states that given a sample of independent and identically distributed values, the sample mean converges to the true mean.

The following is an interaction between a user and an AI assistant that is related to the above text.

[[[User]]]  Jacqueline is a statistician. She is currently analyzing a large amount of data. She is applying the law of large numbers in order to get some conclusions. In order to better understand the law of large numbers, she is considering two samples A and B. Sample A is not independent but sample B is independent and identically distributed.
Which sample's mean does not converge to the true mean, if any, sample A or sample B?
[[[Assistant]]] The law of large numbers (LLN) is a significant concept in probability theory. It suggests that the average of the results obtained from a large number of independent and identical random samples will converge to the true value, if it exists. This means that the sample mean will converge to the true mean.

Now, let's consider Jacqueline's samples. She has sample A which is not independent and sample B which is independent and identically distributed. According to the conditions stated in the LLN, only samples that are independent and identically distributed can make their means converge to the true mean.

Since sample A is not independent, its values are likely related or influenced by previous observations. This lack of independence can skew the mean and prevent it from converging to the true mean. On the other hand, sample B satisfies the condition of independence and identical distribution. Therefore, its mean has a higher chance of converging to the true mean.

Hence, if we are looking for a sample whose mean does not converge to the true mean, based on the given information, it would be sample A.
```

Zinc Deficiency
```
[[[Title]]] Zinc deficiency
[[[Content]]] Zinc deficiency is defined either as insufficient zinc to meet the needs of the body, or as a serum zinc level below the normal range. However, since a decrease in the serum concentration is only detectable after long-term or severe depletion, serum zinc is not a reliable biomarker for zinc status.[1] Common symptoms include increased rates of diarrhea. Zinc deficiency affects the skin and gastrointestinal tract; brain and central nervous system, immune, skeletal, and reproductive systems.

Zinc deficiency in humans is caused by reduced dietary intake, inadequate absorption, increased loss, or increased body system use. The most common cause is reduced dietary intake. In the U.S., the Recommended Dietary Allowance (RDA) is 8 mg/day for women and 11 mg/day for men.[2]

The following is an interaction between a user and an AI assistant that is related to the above text.

[[[User]]]  Two groups of children were studied over a period of 6 months. Group Z had 10 children who were given zinc supplements so they would not develop from lack of this mineral. Group J had 10 similar children who were not given any supplements or pills to take, so they could be part of a control group.
Which group of children would less likely have increased rates of diarrhea?
[[[Assistant]]] Zinc deficiency has been established to lead to increased rates of diarrhea. Since zinc aids in the regulation of the gastrointestinal tract, its lack can cause digestive system issues such as diarrhea. Therefore, preventing zinc deficiency would logically reduce the occurrence of diarrhea. In the given scenario, Group Z was supplied with zinc supplements to avoid zinc deficiency, while Group J served as a control and received no supplements. Thus, with the body's requirements for zinc being met through the supplements in Group Z, one would expect that they would less likely experience the symptoms of zinc deficiency, including increased rates of diarrhea. So, the group of children less likely to have increased rates of diarrhea are Group Z.

```

Cortado
```
[[[Title]]] Cortado
[[[Content]]] A cortado is a Spanish beverage consisting of espresso mixed with a roughly equal amount of warm milk to reduce the acidity,[1][2] although the exact ratios have considerable regional variation.[3] The milk in a cortado is steamed, but not frothy and "texturized" as in many Italian coffee drinks.[4] The cortado is commonly served all over Spain.[5] The word cortado is the past participle of the Spanish verb cortar (to cut), in the sense of "dilute", and can refer variously to either coffee or espresso drinks throughout Spanish and Portuguese speaking countries.

The following is an interaction between a user and an AI assistant that is related to the above text.

[[[User]]]  Brett and Sean land good jobs in Madrid and Lisbon respectively. As a result, they both have a good income and can afford to develop a taste for the fine things in life. Being in Europe they both develop a taste for cortados. Sean, however, stays with his job in Lisbon, while Brett moves back to Seattle and tries to find a job. Brett is unsuccessful and moves back in with his parents.
Which of the two, Sean or Brett, will have an easier time finding a barista that knows what a cortado is?
[[[Assistant]]] A cortado is a type of beverage that originated in Spain. It consists of espresso mixed with warm milk, which reduces the acidity of the coffee. The milk in a cortado is steamed, but not to the point of being frothy or texturized, as is often the case in many Italian coffee drinks. The cortado is a common beverage all over Spain. The word "cortado" comes from the Spanish verb "cortar", which means "to cut" or "dilute". This term can refer to either coffee or espresso drinks throughout Spanish and Portuguese speaking countries.

Given that Brett has moved from Madrid, where he was accustomed to enjoying cortados, to Seattle, his chances of finding a barista who knows what a cortado is will likely be more difficult. In contrast, Sean, who has remained in Lisbon where cortados are also popular, will have an easier time finding a barista who is familiar with this type of beverage.

Therefore, based on their respective locations, Sean will have an easier time finding a barista that knows what a cortado is compared to Brett.```

```