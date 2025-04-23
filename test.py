import re

string = """<｜begin▁of▁sentence｜>
        You are given a context document and a related question. Your task is to generate a comprehensive answer based on the context.

        Context:
        Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1536–1547November 16 - 20, 2020. c⃝2020 Association for Computational Linguistics1536CodeBERT:A Pre-Trained Model for Programming and Natural LanguagesZhangyin Feng1∗, Daya Guo2∗, Duyu Tang3, Nan Duan3, Xiaocheng Feng1Ming Gong4, Linjun Shou4, Bing Qin1, Ting Liu1, Daxin Jiang4, Ming Zhou31 Research Center for Social Computing and Information Retrieval, Harbin Institute of Technology, China2 The School of Data and Computer Science, Sun Yat-sen University, China3 Microsoft Research Asia, Beijing, China4 Microsoft Search Technology Center Asia, Beijing, China{zyfeng,xcfeng,qinb,tliu}@ir.hit.edu.cnguody5@mail2.sysu.edu.cn{dutang,nanduan,migon,lisho,djiang,mingzhou}@microsoft.comAbstractWe present CodeBERT, a bimodal pre-trainedmodel for programming language (PL) andnatural language (NL). CodeBERT learnsgeneral-purpose representations that supportdownstream NL-PL applications such as nat-ural language code search, code documen-tation generation, etc.We develop Code-BERT with Transformer-based neural architec-ture, and train it with a hybrid objective func-tion that incorporates the pre-training task ofreplaced token detection, which is to detectplausible alternatives sampled from generators.This enables us to utilize both “bimodal” dataof NL-PL pairs and “unimodal” data, wherethe former provides input tokens for modeltraining while the latter helps to learn bet-ter generators.We evaluate CodeBERT ontwo NL-PL applications by ﬁne-tuning modelparameters.Results show that CodeBERTachieves state-of-the-art performance on bothnatural language code search and code docu-mentation generation. Furthermore, to inves-tigate what type of knowledge is learned inCodeBERT, we construct a dataset for NL-PLprobing, and evaluate in a zero-shot settingwhere parameters of pre-trained models areﬁxed. Results show that CodeBERT performsbetter than previous pre-trained models on NL-PL probing.11IntroductionLarge pre-t...

        Question:
        Are LIME and Alvarez-Melis and Jaakkola (2017) methods dependent on model properties?

        Instructions:
        - Answer based only on the given context if it's relevant.
        - If the context is insufficient or empty, provide the best answer using your own knowledge.
        - Based on the context above, explain your answer in complete sentences.
        - Ensure your answer is:
        1. Directly relevant
        2. Accurate and fact-based
        3. Complete and informative
        4. Clear and well-structured

        Please answer in a full sentence without using boxed notation or letter choices.:
         Answer:
</think>

Yes, the LIME and Alvarez-Melis and Jaakkola (2017) methods are dependent on the model properties.<｜end▁of▁sentence｜>
"""
cleaned = re.split(r"Answer:*?", string)
cleaned2 = re.split(r"</think>", cleaned[2])
cleaned3 = re.split(r"<｜end▁of▁sentence｜>", cleaned2[1])
print(cleaned3[0].split('\n')[-1])