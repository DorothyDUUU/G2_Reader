qa_prompt_template = """
Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}

Format your response as follows: "<answer>the correct answer here</answer>". No additional explanation is needed for your answer.
"""

qa_reason_prompt_template = """
Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}

Format your response as follows: "<reason>detailed reason for your answer here</reason><answer>the correct answer here</answer>". 
Please make sure that your answer is comprehensive and covers all the important information related to the question.
"""

qa_reason_visual_prompt_template = """
Please read the following text and the attached images and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}

Format your response as follows: "<reason>detailed reason for your answer here</reason><answer>the correct answer here</answer>". 
Please make sure that your answer is comprehensive and covers all the important information related to the question. 
Meanwhile, if directly relevant information is not provided, you should try your best to give an answer based on the available context (without mentioning that you do not have the information).
"""


qa_debate_prompt_template = """
Here is a question related to an article: 
{question}

Here is your previous response:
{previous_response}

Here are responses from two other models:
model1: {model1}
model2: {model2}

Please update your response based on the above information. If you think the previous response is correct, you can keep it.
Format your response as follows: "<reason>detailed reason for your answer here</reason><answer>the correct answer here</answer>". 
Also note that the other models are given different context than yours, so be critical regarding whether you or the other models miss any important information.
Please make sure that your answer is comprehensive and covers all the important information related to the question.
"""

qa_critic_prompt_template = """
Here is a question: {question}

The question is accompanied by a relevant article, which is omitted here. 

Here are responses from three different models:
model1: {model1}
model2: {model2}
model3: {model3}

Please provide a critical analysis of the above responses, and decide what is the correct answer. 
Format your response as follows: "<reason>the reason for your judgement</reason><answer>the correct answer here</answer>". 
Note that the majority is not necessarily correct. You can also combine the answers from the three models to form your own answer.
Importantly, the three models are given different parts of the context, so carefully analyze which of their answers is best grounded in the most relevant context.
Please make sure that your answer is comprehensive and covers all the important information related to the question.
"""


import os
# from transformers import AutoTokenizer
import tiktoken
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
base_url = "http://localhost:30030/v1"
api_key = "sk-MBxi1GmhreGlICXz2rYDdbPbgvo1YO6keN7hhMn1Yz8Mlv9d"
client = OpenAI(api_key=api_key, base_url=base_url)
from llm_client import get_default_model, get_openai_client
# get deepseek-r1 tokenizer (for truncation)
tokenizer= tiktoken.encoding_for_model("gpt-4o-2024-11-20")
max_len = 40000 # 65500 # 65536
max_char = 262000 # 262144

# OpenAI-compatible endpoint (default: local vLLM/OpenAI server on 32000)
# client = get_openai_client()
DEFAULT_MODEL = get_default_model()

def qa_reason(prompt, model=DEFAULT_MODEL, temperature=0.5):
    input_ids = tokenizer.encode(prompt, disallowed_special=())
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    if len(prompt) > max_char:
        prompt = prompt[:max_char//2] + prompt[-max_char//2:]

    print("begin generating response")
    response = client.chat.completions.create(model=model,
                                              messages=[{"role":"user", "content":prompt}],
                                              temperature=temperature)
    response = response.choices[0].message.content
    if "</think>" in response:
        response = response.split("</think>")[-1]
    return response


def amem_qa_visual(texts, images, question, model=DEFAULT_MODEL, temperature=0):
    prompt = qa_reason_visual_prompt_template.format(question = question, context = texts)
    user_content = [{"type":"text","text": prompt}]
    if len(images) > 0:
        for image in images:
            user_content.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=35048
        )
    return {
        "answer": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }

def amem_qa_textual(texts, question, model=DEFAULT_MODEL, temperature=0):
    prompt = qa_reason_prompt_template.format(question = question, context = texts)
    user_content = [{"type":"text","text": prompt}]
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=35048
        )
    return response.choices[0].message.content


planner_prompt_template = """
Here is a question: {question}

Here is a long context related to the question. The context is split into several chunks, each wrapped in <chunk_i> and </chunk_i> tags, where i is the index of the chunk.
{chunks}

Please identify the most relevant chunks of the context to answer the question. These chunks should be divided into up to 3 sets. \
Each set will be given to a different model to read more carefully and answer the question.

Format your response as follows: "<set_1>the indices of the chunks in this set, separated by commas</set_1> <set_2>the indices of the chunks in this set, separated by commas</set_2> <set_3>the indices of the chunks in this set, separated by commas</set_3>". 
Here is an example:
<set_1>1, 2, 3</set_1> <set_2>4, 5, 6</set_2> <set_3>7, 8, 9</set_3>. 
No verbal explanation is needed. Note that the number of sets can be less than 3 if you think it is not necessary. \
The number of chunks in each set is at your discretion.
"""

def planner(question, chunks, model=DEFAULT_MODEL, temperature=0):
    prompt = planner_prompt_template.format(question=question, chunks=chunks)
    # input_ids = tokenizer.encode(prompt)
    # if len(input_ids) > max_len:
    #     input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
    #     prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    response = client.chat.completions.create(model=model,
                                              messages=[{"role":"user", "content":prompt}],
                                              temperature=temperature)
    response = response.choices[0].message.content
    # use regex to extract indices
    pattern = r'<set_(\d+)>([\d,\s]+)</set_\1>'
    matches = re.findall(pattern, response)
    
    indices = {}
    for set_num, indices_str in matches:
        indices[int(set_num)] = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip()]

    return list(indices.values())


aggregation_prompt_template = """
Here is a question: {question}

You have been provided with a set of responses from different models. Each model is provided with a different part of the long context related to the question.
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the soundness of each response, \
recognizing that it might be biased, incomplete, or incorrect based on the provided context. Your response should not simply replicate the given answers, \
but should offer a comprehensive and accurate analysis of the answers provided by the different models and give a final answer that adheres to the highest standards of accuracy and reliability.

Responses from models, and the context they are given: 
{responses}

Format your response as follows: "<reason>the reason for your judgement</reason><answer>the correct answer here</answer>". 
Please make sure that your answer is comprehensive and covers all the important information related to the question.
"""

def aggregation(question, responses, model=DEFAULT_MODEL, temperature=0.5):
    prompt = aggregation_prompt_template.format(question=question, responses=responses)
    response = client.chat.completions.create(model=model,
                                              messages=[{"role":"user", "content":prompt}],
                                              temperature=temperature)
    response = response.choices[0].message.content
    if "</think>" in response:
        response = response.split("</think>")[-1]
    return response

# evaluate_prompt_template = """
# Here is a question: {question}

# Here is a list of gold answers annotated by human annotators:
# {gold_answers}

# Here is the answer provided by an AI assistant:
# {assistant_answer}

# We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answers. Please use the following listed aspects and their descriptions as evaluation criteria:
#     - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answers; The numerical value and order need to be accurate, and there should be no hallucinations.
#     - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
# Please rate whether this answer is suitable for the question. Please note that the gold answers can be considered as correct answers to the question.

# The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
# Please note that if the assistant's answer meets the above criteria, its overall rating should be the full marks (100).
# Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
# Then, output a line indicating the score of the Assistant.

# PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
# <start output>
# Evaluation evidence: your evluation explanation here, no more than 100 words
# Rating: [[score]]
# <end output>

# Now, start your evaluation:
# """

evaluate_prompt_template = """
Here is a question: {question}

Here is a gold answer annotated by human annotators:
{gold_answers}

Here is the answer provided by an AI assistant:
{assistant_answer}

We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answers. Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answers; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that the gold answers can be considered as correct answers to the question.

The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's answer meets the above criteria, its overall rating should be the full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:
"""

def evaluate(question, gold_answers, assistant_answer, model=DEFAULT_MODEL, temperature=0):
    prompt = evaluate_prompt_template.format(question=question, gold_answers=gold_answers, assistant_answer=assistant_answer)
    response = client.chat.completions.create(model=model,
                                              messages=[{"role":"user", "content":prompt}],
                                              temperature=temperature)
    response = response.choices[0].message.content
    # use regex to extract score
    pattern = r'Rating: \[\[(\d+)\]\]'
    matches = re.findall(pattern, response)
    score = int(matches[0])
    return score


def embedding(documents, model="text-embedding-3-small"):
    # generate in parallel using multithreading
    responses = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(client.embeddings.create, model=model, input=doc): i for i, doc in enumerate(documents)}
        for future in as_completed(futures):
            i = futures[future]
            response = future.result()
            responses[i] = response
    return [responses[i].data[0].embedding for i in range(len(documents))]
