---
title: "AI and the data center, part 1"
date: 2026-09-11
authors:
  - name: Justin Lin
    email: justinlin977@gmail.com
    url: https://coffeehouseee.github.io
description: Data centers, something that AI runs on, have plagued the environment for the past couple of years. What happens regarding it?
tags:
  - Ethics
  - Political Science
keywords:
  - Blog post
  - Ethics
---

*Dear reader: I was originally going to make this one blog post. However, there is so much nuance to this topic that I felt a multi-part series would be more appropriate to fit a blog post of this scope. I expect to finish this in my next blog post!*

Happy Friday afternoon! The weather here in Illinois is pretty unbearable right now—80s and 90s day and night. It’s clear that us humans have made quite a toll on the environment, and it doesn’t seem like we’re slowing down.

That’s probably not a good thing in retrospect. The [Climate Clock](https://climateclock.world) currently estimates that we are less than three years away from doing irreversible damage to the planet unless bold change is enacted before the clock ticks to zero. Yet, there still is some hope: the use of renewable energy as fuel has [skyrocketed](https://www.msn.com/en-us/news/other/us-renewable-energy-tops-30-of-the-grid-as-coal-continues-to-slide/ar-AA2bCqKQ) not just domestically, but also internationally, and it’s set to double by 2030. With that, maybe an increase in the role artificial intelligence plays in managing renewable energy sources may be imminent—it’s already on the rise.

However, just as helpful artificial intelligence can be in our environment could it also be dangerous. Generative AI, being immensely energy-intensive, has led to the rise of data centers, a physical room used for storing and managing information. Since artificial intelligence runs on multitudes of data in order for its models to make reasonable judgments like deducting what word to say next or what color a stable diffusion-generated image should, an increase in data centers has been imminent.

Yet, similar to Generative AI, data centers are also incredibly energy-intensive. As the processing systems in a data center compute so much in such little time, they are extremely prone to heating up. Thus, major firms have used water to cool down their computers, which directly takes water away from the city that the data center is used in and renders it unusable for them to utilize. Not only that, but the sheer size of contemporary data centers often use an unsustainable amount of electricity that makes it hard for them to coexist with everyday citizens.

This has made data centers a center of discussion around domestic and international politics, marking a line in a growing division between proponents and opponents of the effects that Generative AI offers to our environment. Just yesterday, U.S. President Donald Trump [firmly backed data centers](https://abc7chicago.com/story/donald-trump-news-today-president-stands-firmly-behind-controversial-data-centers-saying-let-reign/19770876/?userab=kfsn_content_recs-582*variant_b_trending_kfsn-2513,kabc_content_recs-577*variant_a_control-2480,wls_content_recs-584*variant_b_trending_wls-2517,wtvd_content_recs-585*variant_a_control_wtvd-2518,ktrk_content_recs-587*variant_a_control_ktrk-2522,wpvi_content_recs-586*variant_a_control_wpvi-2520,otv_web_content_rec-539*variant_c_trending-2268,otv_web_topic-580*variant_b_topic_redesign-2509,follow_abcnews_otv_on_google-654*variant_a_control-3493,otv_search_page_design_unification-546*variant_b_search_redesign-2300,vertical_video_web-519*control-2163,abcn_popular_reads_exp_followup-648*variant_b_popularityonly-3457,otv_mobile_web_content_recs-639*variant_b_mobile_content_recs_enabled-3431,kgo_content_recs-583*variant_a_control_kgo-2514,otvlink_abcnnav-576*variant_b_local-2479) in a statement encouraging everyone to “let data reign.” It doesn’t seem like the working class feels the same way.

The popularity of data centers has [entered record lows](https://www.cnbc.com/2026/09/06/ai-data-centers-are-transforming-rural-land-markets-fueling-backlash.html) over the past year. Protests [erupt around local city councils](https://www.theguardian.com/us-news/ng-interactive/2026/aug/30/data-center-politics-democrats-republicans) debating whether they should implement one. Newer albums implement a more jagged, imperfect sound in response to the artificial seamlessness found in the [rise of AI-generated music](https://www.media.mit.edu/posts/what-listeners-feel-vs-what-they-say-rethinking-emotional-impact-in-ai-generated-music/). Polls show that everyday Americans [overwhelmingly reject](https://www.forbes.com/sites/saradorn/2026/09/07/nearly-70-oppose-data-centers-poll-finds-as-trump-blasts-communities-that-dont-want-them/) the idea of data centers.

Yet, Wall Street investors and politicians still eagerly invest in the infrastructure due to the sheer profitability of generative artificial intelligence that needs data centers in order to survive. The climate clock must be rolling in its misery right now.

# The six stages of AI

We first have to note that data centers cater towards one of three styles of artificial intelligence currently being researched: Generative AI. But first, we have to get even broader.

## Narrow vs. general vs. super AI

The age of artificial intelligence we are currently living in utilizes models that fit in the classification of narrow (or weak) AI—artificial intelligence models that specialize in one specific task such as reason with humans, predict the outcome of an event, or perform mundane tasks. Despite being able to simulate intelligence when interacting with humans, it is a facade; they communicate and predict based off of algorithms and parameters that are fine-tuned into them during training.

Despite narrow AI currently not being able to develop a sense of sentience or reason at a level above human intelligence, it is extremely capable of simulating human reasoning—in a [previous blog post](https://coffeehouseee.github.io/posts/when-an-ai-model-escapes-what-happens/) that I authored, I explained that a GPT model (not the one that performed that July 2026 escape onto Hugging Face!) had successfully passed the Turing Test, meaning they technically are able to reason with humans at a rate that matches or even exceeds that of humans.

However, narrow artificial intelligence can only perform so much. The next two stages of artificial intelligence mentioned here not only outperform the likes of narrow AI, but are also theoretical. General AI (or [artificial general intelligence](https://en.wikipedia.org/wiki/Artificial_general_intelligence); AGI) is an umbrella of artificial intelligence that can, essentially, perform all the tasks an average human can do. Rather than being a specialist in one task like narrow AI, artificial general intelligence can transfer its skills across multiple fields at once without retraining.

You’ll be sure to see a lot of references to AGI soon in the media; today, OpenAI released yet another GPT model that was immediately followed by celebrations by Nvidia CEO Jensen Huang announcing that “AGI has officially arrived.” The model, GPT-6 Astra, has ironically passed the same cybersecurity benchmark—ExploitGym—that Sol escaped from with a 100% score.

Though, the best way to determine if a model is truly an AGI is if it can do all the stuff a human can do. Even though GPT-6 Astra has only seen a limited release to major corporations, none of OpenAI’s recent work has convinced me that the improvement that they’ve made with Astra would be so momentous to the point where we can actually see the inception of the first general AI model. Needless to say, Huang may be doing this as a marketing ploy, but I digress.

Even higher from AGI is ASI, or artificial superintelligence (or super AI, or any other variation of ASI that there is. If an AGI model is truly on the rise, then the path to ASI looks incredibly straightforward: artificial superintelligence is essentially a form of artificial general intelligence where the model outperforms even the most gifted of humans in tasks (rather than perform at the standard of an average person).

The idea was that if an AGI was capable of self-improvement, then they could continue enhancing its intelligence until it reached a state of ASI. Safe to say, we likely won’t see an ASI model happen for a long time. However, developing models that fit within the mold of AGI or ASI has been a major goal for firms such as that of OpenAI, Google, and Meta.

That’s caused concerns over the public and ethics forums over whether creating a general AI model is really the best thing to do. The worst that could happen is an existential crisis—if the model turns to malice!

## Generative vs. analytical vs. agentic AI

Enough about AGI and ASI; maybe I’ll do another blog post about that for another time. We’re here to talk about data centers on existing artificial intelligence models. For now, all of our models (except for the rather contentious GPT-6 Astra, though the likelihood of that being AGI is pretty small) exist within narrow AI. Within the umbrella of narrow artificial intelligence, there are three branches: generative, predictive, and agentic.

I lied. Generative AI lies within the branch of predictive artificial intelligence. I’ll explain. Stable diffusion—the process within models that allows for them to “generate” images—uses a series of parameters for them to predict exactly what color each pixel in the image should gravitate to. The image that they use actually starts out as pure noise (that’s why early AI-generated images were grainy, for lack of a better term) and slowly clears out until each pixel is within the accepted range of the model in image generation.

The same goes for LLMs: no artificial intelligence model used today actually comprehends the English language, but they speak it by ranking certain words to use and deducing based on certain parameters that were internally fine-tuned which words are most appropriate to use. Both LLMs, stable diffusion models, and anything else that falls under the scope of generative AI use a predictive model to make the thing that’s being generated.

Yet, the newly-minted generative AI uses a far greater amount of energy compared to other non-generative models that also are predictive. Take, for instance, the analytical artificial intelligence models that have been used in every other facet of life that isn’t a major AI firm—the ones used to detect from a brain wave if someone is showing signs of cancer, or deciding where on an island is the best place to [place a wind turbine](https://www.sciencedirect.com/science/article/pii/S2772783125000494) (this was the first ML paper I ever read, by the way!).

It seems without a doubt that the predictive models used to optimize everyday scenarios like that use far less parameters and training data than—after all, the task that they are performing is much simpler than that of speaking and drawing like a human. Compare that to the likes of generative AI, which needs to decipher the prompt given to them, know how to execute the prompt, and make a high-quality image that gives its best ability to follow that prompt to the tea. Now, multiply that by scales of millions.

I would go on a side tangent about how unnecessarily “dumb”—for lack of a better term—generative AI is when it comes to the world of creativity. Unfortunately, I will begrudgingly digress. However, the fact that millions of people at a time are using artificial intelligence and stable diffusion to make images, videos, essays, or even have a conversation with, is quite wasteful in retrospect when looking at how much data and computing power is needed.

# Hark, the data center!

That’s where the data center comes in. Most of the general population probably has a good grasp on what a data center actually is: a room with computers used to store and manage data. Most of the general population should also know that for the past half-decade, the largest tenants and occupants of data centers are artificial intelligence firms who need it to store their excess data that they use to train their flagship models.

This is where we have to go back to our definitions of the multiple types of AI there is. Most artificial intelligence models do not run on the likes of the data center. The majority of models used in our lives are often infinitesimally small in comparison to the large generative AI models you observe all over the internet—they often run on extremely small amounts of data (to fit the niche problem the model aims to solve) and the number of parameters they use are much smaller, as the diminishing returns on these models occur much faster.

Not to mention, I haven’t even gotten started on agentic AI yet! Large firms like OpenAI, Google, Microsoft, and even our good friend Anthropic (with their Claude Code program) have already developed their own versions of an agentic artificial intelligence model that plan and execute a plan based on a single prompt.

You have very likely already seen this play out already. With the rise of vibe coding and the inception of platforms such as base44 and Claude Code, agentic AI platforms are quickly becoming the new norm of a sort of convenience artificial intelligence creates for us humans. Though—yes—the idea of a model planning and executing tasks on your ideas sounds like an amazing proposal to make to the public, the planet would like anything other than that.

Models utilizing agentic artificial intelligence use an extreme amount of energy, roughly [six hundred more times](https://www.theclimatebrink.com/p/the-real-energy-use-of-agentic-ai) that of an actual executed prompt from an LLM model. When again, put to a scale of millions of people using this technology on a daily basis, energy demands skyrocket. By 2030, artificial intelligence-led data centers will be projected to use about [an eighth of electricity](https://eta.lbl.gov/publications/united-states-data-center-energy-2025) in the United States.

## What’s so bad about that?

Enter our first problem. Donating an eighth of our electricity infrastructure simply to the demands of artificial intelligence models requires systems we don’t currently have—especially when pitted against the needs of the rural towns that have the land contractors need for a data center. A small household of five uses around [ten thousand kilowatt hours](https://www.eia.gov/tools/faqs/faq.php?id=97&t=3) (kWh) a year, which leads us to conclude that a small town of around five thousand people likely uses around ten million kWh/yr, or around ten thousand megawatt hours (MWh) a year.

Compare that with a data center, whose median uses [8.7 million kWh per year](https://www.datacenter-asia.com/blog/how-much-power-does-a-data-center-use/). Not to mention, larger data centers and supermassive campuses whose construction have been on the rise can use from 870 million kWh a year to numbers as large as one gigawatt at all times—that is, one million kWh is in circulation twenty-four-seven with these large data centers. Numbers can easily pile up to the billions with those rooms.

They sure have. In 2024, domestic data centers have consumed up to [183 terawatt hours (TWh)](https://www.pewresearch.org/short-reads/2025/10/24/what-we-know-about-energy-use-at-us-data-centers-amid-the-ai-boom/) of energy, which roughly translates to 183 billion kWh. Yet, that statistic, which concludes that data centers in the United States used up more energy than the entire country of Pakistan, was conjured in the early stages of the AI boom—I would not be surprised to see numbers even larger in the present or the future.

The authors of the Pew Research article that I cited in the above paragraph agree with me. They predict a more-than-double increase of energy consumption from data centers as we head into the new decade, with their estimate being at 426 TWh.

Yet, rural communities are already paying the cost that data centers have given to them. Because a single data center uses enough energy to power [one hundred thousand homes](https://www.lincolninst.edu/publications/land-lines-magazine/articles/land-water-impacts-data-centers/), the towns and the land that the data center is built on are often exhausted from energy resources upon the completion of a center. One anecdote explains a town in California whose energy supplier [ditched powering their town](https://arstechnica.com/ai/2026/05/energy-supplier-abandons-lake-tahoe-residents-to-serve-data-centers/) in favor of building more data centers. Another saw residents of a town [paying electricity bills](https://www.consumerreports.org/data-centers/ai-data-centers-impact-on-electric-bills-water-and-more-a1040338678/) triple their regular quote.

Electricity isn’t the only utility that rural citizens have to worry about in the wake of a data center. A mid-sized data center that uses around 8.7 kWh a year also consumes up to [300,000 gallons of water a day](https://www.fwpcoa.org/content.aspx?page_id=5&club_id=859275&item_id=130961). This large amount of consumption primarily comes from the fact that such great values of electricity make the computers holding the data extremely prone to temperature; thus, water is used in circulation to cool the systems down.

We now have another problem—the water that data centers use to cool the computers down renders itself unusable after circulating through a room. Even worse, most data centers worldwide still use fresh water for cooling, directly taking precious resources away from people in surrounding towns and areas who need it most. The chemical buildup in the water after circulating through a data center is simply too great to make it fit for human consumption.

There have been strides to fix this problem ever since this problem has entered mainstream media as a negative to data centers. Now, most data centers reuse their waters before disposing of them, but not to such an extent to the point where our water problem is fixed. Similarly, new projects are straying away from exclusively using freshwater to cool down their data centers, but that doesn’t stop the hundreds of existing ones who still utilize freshwater.

The problems surrounding data centers and their usage of water are already hurting American citizens. You know the stories: citizens living close to a data center experience murky, brown water reminiscent of Flint, Michigan. Faucets in said citizens’ homes barely work, and sputter out dirty liquid as the neighboring towns’ water supply becomes depleted from the demands of the data center.

Not to mention the [light pollution](https://darksky.org/news/data-center-light-pollution-darkskys-statement-on-responsible-development/) that comes with developing a data center, whose 24/7 security surveillance systems and exterior illumination hurt neighboring ecosystems from thriving and hurt the very essence of biodiversity in rural America.

Data centers hurt people. It’s imperative people do something about it before it affects the rest of our populace too.

# Going about fixing this

That’s the topic of part two of my blog, which I will post in the coming week. There, we’ll talk about how we can sustainably fix the problems surrounding data centers using policy and other solutional fixes that may help us curb this problem for good. We’ll also discuss the growing role generative AI has in our society, and whether the advent of artificial intelligence is beneficial for our climate as a whole.

See you then!
