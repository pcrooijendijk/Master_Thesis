import re

test = ["DeepSeek Artificial Intelligence Co., Ltd. (deepseek.com) is a Chinese company dedicated to developing and advancing AI technology. The company offers a wide range of AI-related services, including but not limited to, natural language processing, machine learning, computer vision, and robotics. With its strong focus on research and development, deepseek has established itself as a leader in the AI field. For more detailed information, please visit the official website: <https://www.deepseek.com>.<｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜>"]
fin_output = re.search(r"\s*(.*?)<｜end▁of▁sentence｜>", test[0], re.DOTALL)
fin_output = fin_output.group(1).strip()

print(fin_output)