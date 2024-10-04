**September** 19, 2024
[[Argilla]]
HuggingFace Dataset: [Link](https://huggingface.co/datasets/argilla/FinePersonas-v0.1)

An application by [[Argilla]] of the [[Persona Hub]] technique/recipe to the [[FineWeb-Edu]] dataset, using a [[Distilabel]] pipeline.
Open dataset of ==21 million detailed personas for diverse and controllable synthetic text generation==.
- Because these synthetic personas have been grounded on webpages from the fineweb-edu dataset, there's a ==strong bias towards personas in the education and scientific domain.==

An example persona might describe:
> "A network engineer with a focus on routing protocols and preparing for Cisco certification exams, particularly CCNA."

or
>"A historian specializing in medieval English history and the preservation of historical documents"

or
> "A licensed therapist specializing in eating disorder recovery, likely with extensive experience in psychotherapy and a deep understanding of the complex interplay between physical and emotional health in the recovery process."


Personas have "labels" attached to them, like
> \["Timekeeping Professional", "Horology Enthusiast", "Scientific Researcher"\]

or
> \["Occupational Therapy", "Child Development", "Special Education"\]

(It seems to me like these labels all make sense together, so I don't think they've been randomly selected to be combined. Don't recall if that's from the PersonaHub technique or not.)
