import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from ragas.testset import TestsetGenerator
from ragas.evaluation import evaluate
from ragas.metrics import context_precision, answer_relevancy, faithfulness

import phoenix as px
from phoenix.session.evaluation import get_qa_with_reference
from phoenix.trace import SpanEvaluations
from phoenix.trace import using_project

def build_index(storage_path, storage_name, documents, transformations):
    if not os.path.exists(storage_path+str(f'/{storage_name}')):
        index = VectorStoreIndex.from_documents(documents, transformations=transformations, show_progress=True)
        index.storage_context.persist(persist_dir=storage_path+str(f'/{storage_name}')) 
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_path+str(f'/{storage_name}')))

    return index

def generate_testset(test_path, dir_path):
    if os.path.getsize(test_path) > 0:
        df_test = pd.read_csv(test_path, index_col=[0])
    else:
        loader = PyPDFDirectoryLoader(dir_path)
        docs = loader.load()
        generator_llm= ChatOpenAI(model="gpt-4o-mini")
        critic_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
        embeddings = OpenAIEmbeddings()
        generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
        df_test = generator.generate_with_langchain_docs(docs, test_size=20)
        df_test = df_test.to_pandas()
        df_test.to_csv(test_path)

    return df_test

def generate_response(query_engine, df_test):
    test_questions = df_test.question.values
    responses = [generate_answer(query_engine, q) for q in test_questions]

    dataset_dict = {
        "question": test_questions,
        "answer": [response["answer"] for response in responses],
        "contexts": [response["contexts"] for response in responses],
        "ground_truth": df_test.ground_truth.values.tolist(),
    }
    
    ds = Dataset.from_dict(dataset_dict)
    return ds

def generate_answer(query_engine, question):
    response = query_engine.query(question)
    return {
        "answer": response.response,
        "contexts": [c.node.get_content() for c in response.source_nodes],
    }

def evaluation(querying_project_name, query_engine, df_test, model):
    with using_project(querying_project_name):
        responses = generate_response(query_engine, df_test)

    with using_project("evals-ragas"):
        evals = evaluate(dataset=responses, metrics=[context_precision, faithfulness, answer_relevancy], raise_exceptions=False, llm=model)

    return responses, evals

def add_evaluation(client, query_project_name, eval_result):
    eval_result.scores = eval_result.scores.map(replace_nan_with_zero)
    eval_scores = pd.DataFrame(eval_result.scores)
    eval_data_df = pd.DataFrame(eval_result.dataset)
    eval_data_df = eval_data_df.merge(get_spans_questions(client, query_project_name), on="question").set_index("context.span_id")
    eval_scores.index = eval_data_df.index

    for eval_name in eval_scores.columns:
        evals_df = eval_scores[[eval_name]].rename(columns={eval_name: "score"})
        evals = SpanEvaluations(eval_name, evals_df)
        client.log_evaluations(evals)

    return eval_result, eval_scores

def replace_nan_with_zero(example):
    for key in example:
        if isinstance(example[key], float) and np.isnan(example[key]):
            example[key] = 0
    return example

def get_spans_questions(client, querying_project_name):
    spans_dataframe = get_qa_with_reference(client, project_name=querying_project_name)
    span_questions = (
        spans_dataframe[["input"]]
        .sort_values("input")
        .drop_duplicates(subset=["input"], keep="first")
        .reset_index()
        .rename({"input": "question"}, axis=1)
    )
    return span_questions

def create_results(names, results):
    df = pd.DataFrame(columns=list(results[0].keys()))
    rows = []

    for name, result in zip(names, results):
        new_row = pd.Series(result, name=name)
        rows.append(new_row)
    
    df = pd.concat(rows, axis=1).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'name'}, inplace=True)
    
    return df

def plot_aggregate_evaluation(df):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    sns.lineplot(data=df, x='name', y='context_precision', marker='o', color='purple', ax=axs[0])
    sns.lineplot(data=df, x='name', y='faithfulness', marker='o', color='blue', ax=axs[1])
    sns.lineplot(data=df, x='name', y='answer_relevancy', marker='o', color='green', ax=axs[2])
    plt.show()

def plot_individual_evaluation(dfs, column_name):
    num_rows = len(dfs[0])
    for i, df in enumerate(dfs):
        dfs[i] = df.reset_index(drop=True)

    for row_idx in range(num_rows):
        values = [df.iloc[row_idx][column_name] for df in dfs]
        plt.figure(figsize=(6, 1))  
        sns.lineplot(x=range(len(dfs)), y=values, marker='o')
        plt.title(f'Changes in "{column_name}" for question row {row_idx}')
        plt.grid(True)
        plt.show()