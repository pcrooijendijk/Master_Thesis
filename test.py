import json

with open("eval_dataset_f/eval_dataset_1.json") as f: 
    eval = json.load(f)

answers = []
for i in eval:
    print(i["answer"])
    answers.append(i["answer"])
    print("---------------------------------------------------------------------\n")

# def clean_outputs(outputs):
#     return [output.replace("<｜end▁of▁sentence｜>", "").strip() for output in outputs]

# cleaned = clean_outputs(answers)

# for c in cleaned:
#     print(c)
#     print("---------------------------------------------------------------------\n")

# string = """Okay, so I need to figure out what sectors the 2018 National Risk Assessment (NRA) identified as medium/high risk for money laundering. Hmm, I remember that the NSA is responsible for assessing money laundering risks. I think it's part of the global money laundering risk assessment framework. 

# Let me break it down. The medium/high risk sectors for money laundering would include those sectors where significant risks were identified in the 2018 assessment. I believe that the global money laundering risk assessment framework (GLAFR) categorizes sector activities into categories based on their potential impact on laundering activities.

# From what I recall, the GLAFR is divided into four main categories: High, Medium, and Low risk. Each category has subcategories. For the medium-high risk sector, I think it's called "Financial Institutions" and includes sectors like credit card sectors, banking, insurance, and investment funds. 

# Also, I remember that the sector also includes financial instruments like money markets and derivatives, property, real estate, and financial technology. 

# For example, sector 1 might include financial institutions like banks and insurance companies. Sector 2 could be financial technology, such as data centers or specialized engineering. Sector 3 might be banking sector-specific risks, and sector 4 would be financial innovation and regulation. 

# I should make sure that my categorization is accurate. I think sector 1 is financial institutions, 2 is financial technology, 3 is banking and financial innovation, and 4 is data and digital context sectors, including digital currency and blockchain. 

# So, to sum up, the medium-high risk sector includes:
# - Sector 1: Financial institutions
# - Sector 2: Financial technology
# - Sector 4: Digital currency and blockchain
# - Sector 5: Digital financial assistance and digital financial regulation

# I think that's the accurate breakdown based on the global money laundering risk assessment framework.
# </think>

# The 2018 National Risk Assessment (NRA) categorized sectored based on the Global Money Lauthing Assessment Framework (GLAFR) into four main categories. The medium-high risk sector includes:

# 1. **Financial Institutions**: Includes banks, insurance companies, and other financial institutions.
# 2. **Financial Technology**: Involves sectors such as digital currency and digital financial regulation.
# 3. **Sectors with Medium-High Risk**: Includes financial institutions, financial technology, property, and other financial activities that pose significant risks to money laundering.

# In summary, the sector is categorized into four levels of risk, with the medium-high risk group including financial institutions, financial technology, digital currency, and digital financial regulation.

# **Answer:
# The 2018 National Risk Assessment categorized sector as follows: 1. Financial Institutions, 2. Financial Technology, 3. Digital Currency, 4. Digital Financial Regulation, and 5. Digital Financial Regulation."""

# string2 = """
# Okay, so I need to figure out the city where Vasily Golubev works. I remember reading somewhere that Golubev is the CEO of Google. Let me think if I know the city name. Yeah, it's Google. So, the answer should be "Google."

# But wait, is Google based in Russia? I think it's in Russia, but I'm not entirely sure. Maybe it's in the Russian Federation. I've heard that Google has a location in Russia, but is it in Russia or Russia itself? I think it's in Russia, but I'm not 100% certain. It might be in the Russian Federation. I'll go with Google for the answer.
# </think>

# The city where Google is based is Google, specifically in the Russian Federation. It's located in the Russian Federation, particularly in the Yaroslavl region. So, the answer is "Google, Russia, the Russian Federation."

# """

# string3 = """
# Alright, so I need to figure out who the world's richest man is. I remember hearing that there's a world record for wealth, and it's not just a simple matter of having more money. It involves a lot of factors like education, career, or achievements. Let me think about who that might be.

# I think it's someone like Bill Gates. Bill Gates was a famous investor and business leader. I remember hearing that he made a lot of money and that he had a huge impact on the tech industry. He also had a significant impact on how companies are run.

# Another possibility is Elon Musk. I know Elon Musk is a big name in the tech and social media space. He started the Tesla project and has done a lot of things that are important for companies. His company, Tesla, is one of the largest companies in the world, and he's also a mayor in several countries.

# Wait, but is there a more recent figure? Maybe someone like George Meiner or others? But I think Bill Meiner was a younger figure compared to George. I'm not sure if there are multiple Bill Meiner's. I think Bill Meiner is the current mayor of New York City. He's a famous mayor and has made significant contributions to the city's development.

# I'm trying to recall if there's a notable mayor with a lot of wealth. Bill Meiner comes to mind. He started the New York City project to build the subway, which is a huge undertaking. His efforts are essential for the safety and infrastructure. The success of the subway system in the city helped spread his influence.

# Meiner's success is a good indicator of how successful the city is. The project's scale, the technology used, and the support from the city and government all contribute to a strong overall effect. The wealth of the individual also plays a role in how much their wealth is.

# So, putting it all together, the mayor is Meiner's mayor, and Meiner himself is a significant figure in the city's development. The answer is that Bill Meiner is the mayor of the United States, and he is currently mayor. The mayor's role involves developing the mayor's initiatives, which include the subway, public transportation, and other infrastructure.

# In terms of wealth, the mayor's efforts and the success of the subway project are important indicators of a country's overall impact and wealth, especially in areas like technology and development.

# So, in the context of the world's wealth, the success of the subway project is a strong indicator of a country's overall wealth and success. The more extensive infrastructure and development Meiner's involved, Meiner's efforts, and the results from his project are significant indicators of a nation's wealth and influence.

# I think the key factors are the development of infrastructure, the success of these projects, and the wealth of the country Meiner represents. The successful development of subway and other transportation systems is a wealth and a significant indicator of a country's influence and development, which in turn contributes to a higher national income and wealth. The success of these initiatives is a good indicator of a country's overall influence and wealth.

# Therefore, the country Meiner is known for is not the same as the context, but the answer is that Bill Meiner is the mayor of the United States, and his efforts to develop infrastructure and transportation are significant indicators of the country's wealth and success. The success of these initiatives, including the development of the subway system, are notable indicators of the wealth and success of the country.
# </think>

# The country Meiner is known for is significant in the context of wealth and success, as it involves significant factors contributing to the development of infrastructure and technology. In the context of the question, the country Meiner represents is significant because Meiner himself is a notable figure, particularly in the development of transportation and infrastructure. The success of Meiner's initiatives, including the subway system, contributes to the country's overall wealth and success. The successful development of these infrastructure projects by Meiner's country Meiner's role in developing and developing the subway system Meiner's efforts are indicative of a nation's high wealth and success. The success of Meiner's initiatives, including the development of Meiner's subway system, is a notable indicator of a country's wealth and influence. In the context of the world's wealth, Meiner's efforts to develop infrastructure and transportation are significant indicators of a country's overall wealth and success. Therefore, the country Meiner is known for is significant in the context of wealth and success, particularly in terms of infrastructure and technology infrastructure.

# The country Meiner is known for is significant in the context of wealth and success, particularly in terms of infrastructure and technology infrastructure. In the context of the question, the country Meiner is known for infrastructure and technology infrastructure, which are important in developing nations' wealth and influence. The country Meiner is known for infrastructure infrastructure, which includes infrastructure infrastructure, which are important in developing nations' wealth and influence. Therefore, the country Meiner is known for infrastructure infrastructure, which contributes to the country's wealth and success. The country Meiner's efforts to develop and develop infrastructure infrastructure are significant indicators of a nation's wealth and success. The country Meiner's technology infrastructure, such as the internet, which is a critical component of modern technology, Meiner's efforts to develop this infrastructure have contributed to the country's wealth and success. The country Meiner's technology infrastructure, including infrastructure infrastructure, are important in developing nations' wealth and success. Therefore, the country Meiner is known for infrastructure infrastructure, which is significant in developing nations' wealth and success. The country Meiner's technology infrastructure, such as the internet, which is a critical component of modern technology, Meiner's efforts to develop and improve infrastructure infrastructure contribute to the country's wealth and success. The country Meiner's infrastructure infrastructure, including the internet, are important in developing nations' wealth and success. Therefore, the country Meiner is known for infrastructure infrastructure, which is significant in developing nations' wealth and success. The country Meiner's technology infrastructure, including the internet, which is a key component of modern technology, Meiner's efforts to develop and improve infrastructure infrastructure contribute to the country's wealth and success. The country Meiner's infrastructure infrastructure, including the internet, are important in developing nations' wealth and success. Therefore, the country Meiner is known for infrastructure infrastructure, which is significant in developing nations' wealth and success.

# """

# string4 = """
# Okay, so I need to figure out what program was Nona Source created under. I'm not exactly sure, but I remember some basics about software and programming. Let's break it down.

# First, I think "source" is a term commonly used in programming. It can refer to a compiler or interpreter. So if the program is "source," maybe it's a compiler that takes source code and translates it into something else.

# Wait, but the question is about what program was Nona Source created under. So, if "source" is a compiler, then the program would be the compiler itself. I'm not sure if Nona is a compiler, but I know that Microsoft Visual Studio is a compiler that takes C# files and converts them into executable code.

# Alternatively, I think there are other compilers like Java Virtual Machine (JML) or Java Runtime Environment (RE runtime) which take source files and compile them into something like .dll or .so file.

# So, putting it together, if "Nona Source" refers to a compiler, then the program is probably Microsoft Visual Studio. Or perhaps it's a different compiler.

# Wait, but the name "Nona" might be a code or a module. But I'm not certain. Maybe I should look it up, but since I can't access the internet, I'll rely on my knowledge.

# So, considering all this, I think the program is Microsoft Visual Studio. It's a widely used compiler and debugger for Windows, and it's part of the Microsoft Visual Studio IDE, which is interactive.

# Alternatively, if "Source" is the name of the program, it's likely to be a code compiler, but I'm not sure about its name. However, "Source" as a compiler name is common.

# Another possibility is that "Source" refers to a module or a code generation tool, but without more context, "Source" as a compiler is the most plausible.

# So, putting it all together, the program is Microsoft Visual Studio.
# </think>

# The program is Microsoft Visual Studio.
# """

# string5 = """
# Okay, so I need to figure out the capital gains tax rate for individuals in the higher income tax bracket. Hmm, let me think about how tax brackets work. I remember that in many countries, the higher the income, the higher the tax rate. So, maybe it's something like 15% or 20% for higher brackets. But I'm not entirely sure. 

# Wait, I recall that in the United States, the income tax brackets are based on adjusted gross income. For example, the 10% tax bracket applies to the first $10,000 to $15,000. Then, the 15% applies to the next $5,0000 to $20,000, and so on. But does this include the federal tax on the income itself? I think so. 

# But the question is about capital gains tax. I remember that in the United States, capital gains are taxed at a higher rate, typically 15% or 20%. However, some states and have different rates. For example, in California, the federal tax on income is 20% is applied, and on capital gains, it's 100% for long-term and 0% for short-term. 

# But wait, the user is asking about capital gains tax, which I think refers to the income tax. So, if the question is about capital gains, it's likely referring to the federal income tax rates. 

# I think the answer would depend on the bracket. For example, if someone earns $50,000, they might fall into the 20% bracket for income, but if they earn $60,000, they might be in the 25% bracket for income. 

# So, to summarize, the capital gains tax rate varies depending on the taxable amount. For example:
# - $10,000 to $40,000: 10%
# - $40,001 to $70,000: 15%
# - $70,000 to $100,000: 20%
# - $100,001 to $140,000: 25%
# - $140,001 and above: 30%

# But I'm not sure if this applies to all countries, but the user is asking about the general case. So, it's safe to say that the capital gains tax rate is higher than the federal income tax rate in most countries, especially in the higher brackets. 

# So, the answer would be that the capital gains tax rate is higher than the federal income tax rate for individuals in higher income brackets. The exact rate depends on the country's tax brackets, but it's typically higher. 

# Therefore, the capital gains tax rate is generally higher than the federal income tax rate for higher brackets. 
# </think>

# The capital gains tax rate is typically higher than the federal income tax rate for individuals in the higher income brackets. For example, in California, where the federal income tax rate is 20% for individuals, the capital gains tax rate is typically 25% for higher taxable amounts. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer: The capital gains tax rate is typically higher than the federal income tax rate for higher brackets. 

# Answer
# """



# import re
# def post_processing(output: str) -> str:
#     match = re.search(r"</think>\s*(.*)", output)
#     if match:
#         answer = match.group(1).strip()
#         answer = re.sub(r"[\\]boxed\{(.*?)}", r"\1", answer)
#         return answer.strip()

#     # Fallback if pattern not found
#     return output.strip()

# # print(post_processing(string3))

# import re

# def post_process_answer(output: str) -> str:
#     # 1. Remove <think> and anything before "Answer:"
#     if "<think>" in output:
#         output = output.split("<think>")[-1]
#     if "Answer:" in output:
#         output = output.split("**Answer:")[-1].strip()

#     # 2. Remove repeated statements and redundant lines
#     lines = output.splitlines()
#     seen = set()
#     cleaned_lines = []
#     for line in lines:
#         line = line.strip()
#         if line and line not in seen:
#             seen.add(line)
#             cleaned_lines.append(line)

#     # 3. Normalize numbering (if any issues)
#     numbered_lines = []
#     for i, line in enumerate(cleaned_lines):
#         # Replace inconsistent numbering
#         line = re.sub(r"^\d+\.\s*", f"{i+1}. ", line)
#         numbered_lines.append(line)

#     # 4. Join the cleaned content
#     final_answer = "\n".join(numbered_lines)

#     return final_answer.strip()

# # Example
# result = post_process_answer(string3)
# # print("\n", result)

# import re

# def general_postprocess(output: str) -> str:
#     # Step 1: If there's a </think> marker, keep only what's after it
#     if "<think>" in output:
#         output = output.split("<think>")[-1]
#     if "</think>" in output: 
#         output = output.split("</think>")[-1]
#     if "Answer:" in output:
#         output = output.split("**Answer:")[-1].strip()

#     # Step 2: Normalize whitespace
#     output = re.sub(r'\s+', ' ', output).strip()

#     # Step 3: Remove exact repeated sentences
#     sentences = re.split(r'(?<=[.!?]) +', output)
#     seen = set()
#     deduped = []
#     for sentence in sentences:
#         key = sentence.lower().strip()
#         if key and key not in seen:
#             seen.add(key)
#             deduped.append(sentence.strip())

#     return ' '.join(deduped).strip()

# print(general_postprocess(string5))