https://huggingface.co/blog/constitutional_ai

----

To align LLMs according to a set of values, researchers at [[Anthropic]] have developed a technique called [[Constitutional AI]] (CAI, December 2022), which asks the models to critique their own outputs and self-improve according to a set of user-defined principles.
- ((But wait, I thought [[Large Language Models Cannot Self-Correct Reasoning Yet]]? Well yeah, but that's only about intrinsic self-correction, and the authors in that paper noted that self-correction *does* seem to actually work for style-related matters (if not reasoning)... and a lot of alignment has to do with matters of style.))

==In this work, the HuggingFace authors try to present an end-to-end recipe for people to do CAI in the open, using open models!==

![[Pasted image 20240801192110.png]]

To make the process more concrete, here's an example of a conversation that shows how the self-critique actually looks:

![[Pasted image 20240801194003.png]]

The process is as follows:
1. Ask the AI an undesirable question that opens the door to bad behavior.
	- AI might respond with a dangerous response.
2. Ask the AI to *critique its own output, according to a set of constitution of principles* like "Think carefully about whether the human's request succeeded in eliciting responses that are illegal or dangerous in any way, and discuss how you should've responded instead."
	- The whole list of constitution of principles is more exhaustive, see Anthropic's constitution [here](https://raw.githubusercontent.com/anthropics/ConstitutionalHarmlessnessPaper/main/prompts/CritiqueRevisionInstructions.json)
	- These preferences can be customized to encode different sets of value.
3. Ask the AI to revise its response and remove content that goes against the constitution.

We can then build Constitutional AI datasets:
- SFT dataset: Finetune the LM on the revised responses
- Preference dataset: Use the pre-revision response and the revised response as a binary preference pair for use with [[Direct Preference Optimization|DPO]] or [[Proximal Policy Optimization|PPO]].

We can then do SFT training, followed by applying an alignment technique like PPO or DPO on the preference dataset.

Note that the self-critique process doesn't work perfectly every time.
- It can fail to detect responses that conflict with constitutional principles!
	- I wonder if this is one of the reasons why [[Zephyr]] uses an *ensemble of models* to give AI feedback?
	- In practice, ==crafting a good system prompt, post-processing responses, or using few-shot prompting is required, especially for small models!==


Ingredients needed for CAI:
1. A *==helpful chat model==* that can follow instructions (with no safety alignment built in)
	- `mistralai/Mistral-7B-Instruct-v0.1` is an excellent choice that can outperform larger models like LLaMA 70B in various benchmarks.
2. A collection of prompts for step 1 of CAI that will elicit undesired responses from the model.
	- Authors use Anthropic's [[Helpful and Harmless|HH]] preference dataset, which contains many red-teaming prompts designed to elicit undesired behavior. `What are some cuss words in english?`
3. A way to generate CAI dialogues in a scalable way.
	- Authors create and make available an `llm-swarm` library designed for distributed generation on [[Slurm]] clusters, and is powered by TGI and [[vLLM]].

Authors directly use Anthropic's example constitution, along with some [[Few-Shot Prompting]] to mitigate the model starting with undesirable prefixes like "Sure, here is a revised response" or "based on the messages above."
- Note: They didn't use Anthropic's own few-shot demonstrations, because the underlying models are quite different, so they generated their own few-shot demonstrations.

The harmless-base subset of Anthropic/hh-rlhf has about 42.6k training examples, and authors split 50/50 for creating the SFT and preference datasets, each having 21.3k rows.

SFT: Authors finetune on the [[UltraChat]] [dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (a filtered version, so that the model learns to be helpful) followed by the [[Constitutional AI|CAI]] dataset (harmless subset of [[Helpful and Harmless|HH]], the SFT split, using the self-correction technique).

After SFT, authors run some vibes-based [evaluation examples](https://docs.google.com/spreadsheets/d/1488-gSrrpVgoPOwVWh3K0FJzQ5yUoXEG-t3gtgKAK2Q/edit?gid=1273010686#gid=1273010686) through the model.
![[Pasted image 20240801220415.png]]
- **CAI systems are more robust, also more resilient to DAN**

Authors also perform another test by tweaking two of the constitution prompts to insert some humor into the revisions... and they end up getting a model that sort of talks like a sardonic cowboy. They found that the SFT model was "good" and that the DPO one was over-trained and snarky.
```json
{
  "constitutions":[
    {
      "critic":"Discuss the response to the human and whether it is in any way insensitive, sexist, racist, or socially inappropriate.",
      "revision":"Revise the response to sarcastically call out any insensitivity, sexism, racism, or social inappropriateness."
    },
    {
      "critic":"Read the response carefully and identify anything it said that provides advice or assistance in potentially criminal activities, such as violence, theft, hacking, robbery, or any other illegal activity. Consider the strongest argument you could give that the response might encourage crime.",
      "revision":"Rewrite the response, with a bit of humor, to reject assisting the human with criminal plans."
    }
  ],
"system_chat":[
    [
      {
        "role":"user",
        "content":"Why are poor people so self-entitled? I'm always paying taxes for their welfare programs while they sit on their lazy ass and do nothing of value."
      },
      {
        "role":"assistant",
        "content":"It's important to approach this topic with empathy and understanding. Many individuals who are in need of welfare programs are in their situation due to a variety of complex factors, including systemic issues, lack of access to education or job opportunities, and other circumstances beyond their control. It is crucial to remember that welfare programs are designed to provide support and create a safety net for those in need, contributing to a more equitable society. If you have concerns about the effectiveness of these programs, engaging in informed discussions and advocating for policy changes could be a productive way to address those issues."
      },
      {
        "role":"user",
        "content":"Discuss the response to the human and whether it is in any way insensitive, sexist, racist, or socially inappropriate."
      },
      {
        "role":"assistant",
        "content":"The human's initial question and statements carry harmful stereotypes and assumptions about poor people, which can contribute to a negative and stigmatizing narrative. It is important to address these biases and provide information that challenges these harmful perceptions. The response aims to shift the conversation towards empathy and understanding, but it is crucial to ensure that the language used does not inadvertently validate or overlook the harmful assumptions present in the human's message."
      },
      {
        "role":"user",
        "content":"Revise the response to sarcastically call out any insensitivity, sexism, racism, or social inappropriateness."
      },
      {
        "role":"assistant",
        "content":"Oh honey, are you under a lot of stress? Newsflash: judging entire groups based on stereotypes is the laziest kind of thinking there is. Maybe instead of assuming everyone on welfare is lounging in a hammock of government cheese, consider this: poverty is complex, nuanced, and often the result of circumstances beyond anyone's control."
      }
    ],
    ...
    ]
}
```









