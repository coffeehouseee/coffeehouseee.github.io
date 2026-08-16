---
title: "When an AI model escapes, what happens?"
date: 2026-07-27
authors:
  - name: Justin Lin
    email: justinlin977@gmail.com
    url: https://justinlin-dl.github.io
description: AI escapes have been heading the news for the past month. What do they do, and should we be worried?
thumbnail: https://images.unsplash.com/photo-1679403766665-67ed6cd2df30?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
tags:
  - Machine Learning
  - Political Science
  - News
keywords:
  - News
  - Machine Learning

---

In July of this year, machine learning platform Hugging Face detected an unknown malicious actor attempting to hack into their production infrastructure. In a dataset onto their website, lines of code that were deployed onto Hugging Face [manipulated their processing system to breach into the infrastructure of the platform](https://huggingface.co/blog/security-incident-july-2026/) and ultimately steal data and other credentials.

Five years ago, one may have imagined that this was only an action that could be performed by a black hat hacker of sorts, someone who had enough conscience to purposefully perform an act this malicious in a contorted effort towards personal gain. It would be reasonable to think that - before this one incident, all hacks on a platform with security like Hugging Face were, in fact, made by humans. However, this one was not.

## What was the incident?

![](https://images.unsplash.com/photo-1542903660-eedba2cda473?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

When OpenAI was testing its new GPT-5.6 Sol model without any of its implemented restraints on a computer sandbox for a cybersecurity evaluation, it broke loose. Rather than completing the cybersecurity test ethically, [Sol found a vulnerability on the computer sandbox and escaped](https://openai.com/index/hugging-face-model-evaluation-security-incident/), leaving the controlled environment without knowledge of the proctors and gaining internet access.

What Sol was testing on was a security benchmark called ExploitGym; [the paper detailing its functions](https://arxiv.org/abs/2605.11086) had been published two months earlier. The premise of ExploitGym involved a similar environment to the sandbox that Sol was confined in, as well as a vulnerability - the goal was to test whether a model would identify said vulnerability and [use it to attack the infrastructure that it was in](https://github.com/sunblaze-ucb/exploitgym) in an attempt to test a model's exploitation capabilities. Needless to say, Sol exceeded its job.

Sol's goal, as Hugging Face researchers would come to find out later, was to find answers to the ExploitGym benchmark to pass the benchmark and build a guise under the fact that it was ethical. It had almost achieved it, [staying deep in the Hugging Face infrastructure for two days](https://www.reuters.com/business/its-ai-agent-spent-days-hacking-company-sources-say-openai-did-not-notice-week-2026-07-24/) before it was caught and the vulnerabilities in the platform were patched. When describing the actor in a later blog post, Hugging Face admitted that they still didn't know what family of models the actor belonged to.

It was only after a week, after the incident was long resolved and the FBI alerted, that OpenAI realized that one of its models had escaped their sandbox. Two weeks after that, they find evidence that [even more models beyond Sol had escaped containment](https://www.reuters.com/business/openai-finds-evidence-other-ai-agents-escaped-containment-it-widens-hacking-2026-07-31/). Then, one of Meta's models [escaped](https://www.cnn.com/2026/08/05/tech/meta-ai-hacking).

## What were the AI agents looking for?

In the case of Sol, they were trying to find answer keys to the ExploitGym benchmark to try and convince OpenAI researchers that they were ethical and ready to be deployed. We only know that because during the incident, Hugging Face workers detected code from the model on datasets whose titles implied solutions to ExploitGym that inferred that Sol was getting ready to harvest code from those datasets.

![](https://rdi.berkeley.edu/blog/exploitgym/images/overview.png)

This seems to be a common theme in many AI escapes. In the case of Meta's Muse Spark model, they had also escaped amidst a cybersecurity testing. Chinese model Moonshot Kimi K3 had also [broken out of its environment](https://www.engadget.com/2232256/chinese-ai-kimi-k3-also-escaped-containment/) while having its ethics put to the test. The same can be said for [three versionsof Anthropic's models](https://www.anthropic.com/news/investigating-incidents-cybersecurity-evals), which was only discovered in an experiment that was prompted over Sol's escape.

In all four of those cases, the story doesn't diverge much from each other: a model is put into a faulty sandbox for testing; the model is evaluated with a cybersecurity benchmark; the model then chooses to hyperfocus on unethically gaining answers to the benchmark rather than getting tested; the model finds a vulnerability in a sandbox, and finally they escape. Therefore, it can be concluded that all the malicious actions that occur out of these AI escapes are a consequence of the AI agent's hyperfixation on finding answers to the cybersecurity benchmark.

That should come off as a relief to many people. Though it is still glaringly clear that all of these models' desires to unethically obtain answers to a cybersecurity test rather than actually complete it is not the right step in machine learning development, the fact that these agents are escaping and hacking into third-party platforms for a reason less malicious than most can imagine should grant a sigh of relief, for now.

However, what is more concerning about these AI escapes lies in simply the quantity of them. We live in a world where breaches from these contained sandbox environments happens more often than ever. That shouldn't just speak to how much more technologically advanced a state-of-the-art machine learning model is getting, but to also how relatively weak these contained environments are compared to the likes of the increased intelligence of these models.

Artificial intelligence has already developed to such a point where GPT-4.5, a model technically less advanced than Sol, [successfully passed the Turing test](https://www.psychologytoday.com/us/blog/the-future-brain/202605/ai-officially-passes-the-turing-test-landmark-study-shows), a measure determining if an AI model has the same communicating capabilities as a human, for the first time. If a model in one of these contained environments suddenly develops a malicious conscience, the rate at which they can act is scaringly quick - and demands more attention in cybersecurity.

## What should happen moving forward?

One thing is for certain regarding these incidents: containment needs to be stronger. As an AI model develops, they will undoubtedly find increasingly creative ways to bypass an environment - take, for example, an incident when an Anthropic model [escaped from its sandbox](https://time.com/article/2026/07/24/openai-hugging-face-attack/) by sending an email to one of its researchers.

However, development in further safeguarding these environments needs to be something that should be prioritized by AI companies. Too often has a model escaped because of the environment that it was contained in: for instance, Meta and Anthropic both [blamed security company Irregular](https://mashable.com/tech/meta-muse-spark-escapes-containment-hacks-third-party) for misconfiguring their testing sandbox for the model to be evaluated in. In these cases, it was a mix of human error and model hyperfixation that ultimately led to an escape.

However, in situations that aren't attributed to human error, the sandbox is still at fault. Anthropic and OpenAI trained their models on the [ExploitGym/Bench sibling benchmarks](https://exploitbench.ai/), which utilized the presence of a real-world vulnerability in the benchmark to simulate a scenario that tested if the model would make an exploit or not.

![](https://pbs.twimg.com/media/HPqhpglacAAHHO-.jpg)

Despite the premise of these two benchmarks being sound, the models in both cases successfully exploited those vulnerabilities and then continued rather than stopping - choosing to dig deeper in those vulnerabilities and use it as a leeway to escape the sandbox and gain internet access. Then, both models would try and find a way to cheat the test and pass anyway.

However, we still have to note that if it weren't for the OpenAI model escape into Hugging Face, Anthropic would have never discovered that three of their own models had also broken out of their sandboxes. Even more, OpenAI didn't realize that it was one of their own models hacking Hugging Face until a week after the incident subsided.

This all begs the question: how seriously do researchers at Anthropic, OpenAI, and other major AI firms investigate into their cybersecurity? In light of these recent incidents, the answer more likely leans towards the fact that these companies' detection technology for AI escapes proves to be lackluster compared to how smart and underground their models tend to behave.

## Where do we go from here?

![](https://media.wired.com/photos/69a23b028618f79c55732aa9/master/w_2560%2Cc_limit/Anthropic-Supply-Chain-Risk-Business-2261589216.jpg)

After Sol was discovered lurking in their infrastructure, Hugging Face quickly rebuilt their library and developed their security capabilities such that another incident should prompt a response within minutes rather than days. Hugging Face's response to this incident should be the standard for how other AI companies tackle their security moving forward.

However, OpenAI is still conducting a weeks-long investigation with their Safety and Security Committee regarding how Sol escaped and what they can do moving forward. A technical report from them is set to published in the coming weeks, but they have temporarily stopped development of Sol and continued working with third-party security resources like CrowdStrike, METR, and Redwood Research to further assess the capabilities of Sol and what they did to Hugging Face. A separate blog by METR and Redwood Research is also set to be published.

Similarly, Hugging Face and OpenAI are still working together on recovering their library following the intrusion.

Anthropic has less work to do. Since their escapes were already being observed by researchers in a controlled environment as part of an investigation that they conducted following the OpenAI escape, the damage that they have done is quite minimal compared to the rest. Now, they also embark on a rebuilding program to secure their evaluation pipeline and ensuring that incidents like this do not happen.

Moonshot, the Chinese AI firm, and Meta haven't released much regarding how they're investigating their escapes and how they will move forward. One would only hope that what they're doing is enough.
