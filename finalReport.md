# TRAIL Berlin Workshop – Project 7 – Final Report
Explore the gap between existing explainable artificial intelligence techniques and their use as tools

## Introduction

The main objective of our project is to explore the gap between existing explainable artificial intelligence (XAI) techniques and their use as tools to support users’ goals in specific usage contexts. This gap can be problematic for many reasons mainly related to the mismatch between the inner design of technical solutions and the specificities of the end-user (i.e., needs, level of expertise, background, cognitive process, etc.) as well as the specificities of the usage context (i.e., application domain, interaction modalities and constraints, etc.).

To improve the understanding of this gap, we propose to address the following challenge: the design of an explainability tool to help AI researchers and developers improve and debug a multimodal transformer-based model for a classification problem in computer vision. In particular, the design will focus on the presentation of the explanations, with the goal of making them relevant and actionable w.r.t. to the user's goals and usage context.

The project management is based on the [<ins>Design Sprint methodology</ins>](https://www.thesprintbook.com/). This methodology aims to solve problems with a small team and constraint timing. The generic roadmap consist of 5 steps:
1. Make a map & choose a target
2. Sketch competing solutions
3. Decide on the best
4. Build a realistic prototype
5. Test with target customers

Each of these steps involves users in one way or another to ensure that the results are relevant with targeted users’ needs.

Expected benefits of the research:
* The main result is a XAI tool, in the form of a prototype, design to help AI researchers and developers improve and debug a transformer-based model for a classification problem in computer vision. The designed tool, after being made fully functional, would be put on the TRAIL factory.
* By exploring HCI aspects of XAI during the entire design process, we will gain a lot of knowledge about the use of XAI and especially about users’ needs and pitfalls. This knowledge could serve to guide further research and application project in XAI by bridging the gap between XAI techniques and their use.
* The obtained results could also be valorized through a publication later on.
* Finally, we will also demonstrate the relevance of a user-centered approach to design XAI solutions, and more generally to conduct research in XAI.

cf. proposal for a more complete description of Project 7

## User Research & Domain Exploration

![](https://lh3.googleusercontent.com/wFx9fMIDjsoFyNqGJTQ_l4JR7dn-V-mHG6p7ApYF8ANYVzEk6JnpOnUlbEDt8Eo27TCVzsZvpl4JM2bs7yNRzbw_IU2V8GUWNDXxT4If8XBV9KabN2-3j3vCtogyhnixpCMWAHS4BL3zKD-x9xEJkezYlXIWHYMm7cqg5hnoFoaX7sCgpWM)

Final Roadmap with HMW notes and selected targets.

### Roadmap

The first task was to produce a roadmap to guide the work of the sprint by cartographying the addressed problem. The result is shown above

Main elements of the roadmap:
* Target user (blue note): AI researchers/developers (with some knowledge of transformers).
* Beginning of the story (red notes): the target user would like to know if he can trust his model by assessing it globally and/or locally.
* End of the story (green note): the target user gains trust in his model.
* Intermediary steps (yellow notes) from the beginning to the end of the story, i.e., how can the target user achieve more trust in his model

Main steps:
1. The target user specifies his use case.
2. He chooses the most suited method out of a recommended set.
3. He adapts and applies it.
4. He interprets obtained explanations and adapts model/data in consequence.

### Exploratory Interviews

7 exploratory interviews were conducted with AI developers/researchers experienced with transformers. Main takeaways:
* Huge variety of use cases (data modalities, domains, tasks, etc.), it’s impossible to choose one based on representativeness.
* Making a transformer model work is not trivial (finding the right metaparameters & architecture, preprocessing data conveniently, etc.).
* Debugging and local explanations are the most common needs.
* Users prefer to be able to adapt existing tools than to use blackbox tools.
* None of the interviewed users uses multimodal transformers.

### Domain Exploration

To expand our knowledge about transformers, 4 presentations were done by the teams based on literature exploration.

Transformers Survey:
* Application domains & tasks, with a specific focus on [CLIP](https://github.com/openai/CLIP) and [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP)
* Main takeaway: application domains & tasks are very various, it would be hard to be generic and we must choose some use case to address

XAI Generalities: 
* Focus on notions, evaluation and methods
* Slides and notes on demand
* Main takeaway: a lot of literature, importance of user-centered approach

XAI methods for transformers:
* Slides on demand
* Main takeaway: XAI for transformers domain is in its infancy, with no really mature method other than raw attention scores

XAI tools & frameworks:
* Everything that could be useful in our context
* Summary of available tools and frameworks on demand
* Main takeaway: some tools & frameworks exist, but they must be adapted to our needs

### Selected target(s)

To synthesize gathered knowledge and make it actionable, participants were asked to produce HMW (“How Might We…”) notes based on things (features, problems, etc.) they find the most valuable in our context.

After gathering all the notes and voting, these were the most valuable ones:
* HMW propose relevant XAI solutions w.r.t. the huge variety of use cases? (models, data modalities, tasks, etc.)
* HMW help users to choose the most suitable method to theirs needs?
* HMW make a tool with which the user can interact?
* HMW use XAI techniques to help debugging transformer models?
* HMW propose tweakable solutions to address specific needs of different users?

These notes (pink notes on the roadmap) were added to the roadmap to help decide the targets of our sprint (red dots on the roadmap).
* Found the recommended methods for his needs (data, problem, etc.)
* Adapts the recommendations to its implementation
* Choose and interact with explanations

However, only the last two were addressed because of the constraint duration of the workshop.

### User Tests

User tests were initially planned to evaluate our designed prototype. However, since nothing really testable was produced due to various problems, this activity was dropped.

## Technical Proof-of-Concept

### XAI Methods

In our case, explaining the model consists in getting the determining inputs for the task. Available approaches:
* Direct use of attention layers and gradient.
* [Chefer's method](https://openaccess.thecvf.com/content/ICCV2021/papers/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.pdf): chosen because works for all the attentions (co-attention, encoder-decoder and self-attention). By the way, Chefer's method allows to “have a good idea” of which input parts of an image and a text are the most similar.
* Similarity between text and image.

Our work mainly consists in making them more easily adaptable to different models. To do so, we built a wrapper to be able to explain other models than CLIP in the future. We also explored the following data modalities: image/image and text/text. Finally all the experiments were dockerized.

Future works: The ultimate goal is to build a package that is easy to adapt to explain any transformer model with lots of visualisation options.

### Visual Embeddings Check

Visual embeddings check shows that:
* CLIP visual embeddings are quite good and capture both high-level (eyes wide opened vs non-eyes) and low-level details (yellow vs black cat).
* The embeddings similarity difference is small between good/bad match (needs specific adaptation for visualisation).

Visual embeddings can be of practical interest to verify:
* That embeddings are correct (check on the same image and some others),
* If there is a bias (check on several images if there is a difference based on the type of image).

### Visualisation Interface

The interface is inspired from the [VL-Interpret](https://github.com/IntelLabs/VL-InterpreT) tool developed by intel. The tool provide interactive and insightful visualisation enabling the users to see the cross-attention between tokens in a transformers. The tool is made with a python library called [Dash](https://plotly.com/dash/). This was convenient as a lot of team members were already familiar with python.

The VL-interpret tool was dockerized and adapted to be backend agnostic. Thus, the model could easily be replaced by another one. Unfortunately, it was quickly realized that the tool was hard to adapt to generic case. It was decided to scrap the interface and start over from scratch giving us the opportunity to test other visualisation scheme. This part of the project is still a work in progress.

## Publication Potential

Not considered during the workshop. It may be something to discuss afterwards when we have more substantial results.
