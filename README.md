# **Analysis of Arize Phoenix in RAG Systems**  

## **Project Description**  
The project focuses on evaluating the performance of Retrieval Augmented Generation (RAG) systems using the **Arize Phoenix** platform. The goal was to monitor, optimize, and analyze the effectiveness of a question-answering (QA) system by applying various text segmentation techniques, metadata extraction methods, answer generation engines, and reranking models.  

## **System Architecture**  
The system is built on three key components:  
- **LlamaIndex** – a framework for implementing RAG systems,  
- **RAGAS** – a library for generating test sets and evaluation metrics,  
- **Arize Phoenix** – a real-time monitoring platform enabling the analysis of data processing pipelines.  

The input data consisted of 12 PDF files (205 pages) containing information about the Polish-Japanese Academy of Information Technology. To protect privacy, sensitive data was removed.  

## **Verification Procedure**  
1. **Test Set**: 20 queries were generated (including simple, multi-context, and reasoning-based questions) using the **GPT-4o-mini** model.  
2. **Evaluation**: Metrics such as **context precision, faithfulness, and answer relevance** were used.  
3. **Monitoring**: Arize Phoenix was employed to track query processing and response time.  

## **Experiments and Results**  

### **1. Text Segmentation**  
- **Best method**: **Token Splitter** – achieved the highest context precision (95%) with good performance.  
- **Hierarchical Node Parser** – had the shortest response time but lower context accuracy (76%).  

### **2. Metadata Extraction**  
- Enhancing metadata improved **faithfulness** (from 97% to 98%) and **answer relevance** (from 90% to 92%) but slightly reduced context precision (from 95% to 93%).  

### **3. Answer Generation Engines**  
- **SubQuestionQueryEngine** outperformed **HyDE** in speed and response relevance, particularly for reasoning-based queries.  

### **4. Reranking Models**  
- **bge-reranker-large** provided **better context precision** (100%) and reduced response time for reasoning tasks.  

## **Conclusion**  
- **Arize Phoenix** enabled an in-depth analysis of RAG processes and optimization of the QA system.  
- **Best configuration**: Token Splitter, metadata extraction, SubQuestionQueryEngine, and bge-reranker-large.  
- The results indicate that **real-time monitoring is crucial** for identifying and improving the efficiency of RAG systems.  