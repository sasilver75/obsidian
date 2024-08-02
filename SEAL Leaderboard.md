May 29, 2024
[[Scale AI]]
Launch Post: [Scale's SEAL Research Lab Launches Expert-Evaluated LLM Leaderboards](https://scale.com/blog/leaderboard)
Leaderboard Link: [SEAL Leaderboards](https://scale.com/leaderboard)

A set of leaderboards ranking *frontier LLMs* using curated, regularly-updated, private datasets developed by Scale AI's ==Safety, Evaluations, and Alignment Lab ([SEAL](https://scale.com/blog/safety-evaluations-alignment-lab))== and assessed by verified domain experts.

As of August 1, 2024, leaderboards include:
- [Adversarial Robustness](https://scale.com/leaderboard/adversarial_robustness)
- [Coding](https://scale.com/leaderboard/coding)
- [Math](https://scale.com/leaderboard/math)
- [Instruction Following](https://scale.com/leaderboard/instruction_following)
- [Spanish](https://scale.com/leaderboard/spanish)
But Scale plans to continue launching new SEAL leaderboards covering additional domains and capabilities, aiming to refresh these leaderboards multiple times per year.

SEAL Leaderboards offer:
- ==Private Datasets==: Scale's proprietary, private evaluation datasets can't be gamed, ensuring unbiased and uncontaminated results. Datasets will remain private and unpublished.
- ==Evolving Competition==: We periodically update leaderboards with new datasets and models, fostering a dynamic, contest-like environment.
- ==Expert Evaluations==: Our evaluations are performed by thoroughly-vetted experts using domain-specific methodologies, ensuring the highest quality and credibility. Verified (interviewed and tested) domain experts create prompt sets from scratch, and tailor evaluation methods to what we believe works best for that domain.
- ==Data Quality==: Both prompts and ratings have undergone extensive, multi-round reviews and passed internal QA checks; Scale also plans to collaborate with trusted third-party organizations to help review work.

The usual challenges of leaderboards and evaluation include:
- ==Contamination and Overfitting== (Few high-quality datasets have not been contaminated)
- ==Inconsistent Reporting== (Lack of consistency in model comparisons)
- ==Unverified Expertise== (Lack of rigorous assessment of *evaluator's expertise*, e.g. for coding)
- ==Inadequate Tooling== (Lack of good product tooling for understanding and iterating on evaluation results without overfitting to them.)

Scale uses Elo-rankings to compare model performance across datasets; human evaluators compare responses of two models to the same prompt, and rate which is better along a multitude of capabilities. They use the same method as [[ChatBot Arena]], with the Bradley-Terry model... Authors also estimate confidence intervals using a bootstrapping technique (repeatedly sampling from the data with replacement).

