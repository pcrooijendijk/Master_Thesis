from ragas.metrics import RougeScore, BleuScore
from ragas import SingleTurnSample

example = {
        "question": "How many categories of aggression were participants asked to classify texts into?",
        "ground_truth": "3 categories: overt aggression, covert aggression, and non-aggression.",
        "answer": "Participants were asked to classify texts into three categories: overt aggression, covert aggression, and non-aggression. The task aimed to assess their ability to perform a top-level classification of aggression in a language-agnostic manner.",
        "contexts": [
            "The positive response from the community and the great levels of participation in theﬁrst edition of this shared task also highlights the interest in this topic.1IntroductionIn the last decade, with the emergence of an interactive web and especially popular social networkingand social media platforms like Facebook and Twitter, there has been an exponential increase in theuser-generated content being made available over the web. Now any information online has the powerto reach billions of people within a matter of seconds. This has resulted in not only positive exchange ofideas but has also lead to a widespread dissemination of aggressive and potentially harmful content overthe web. While most of the potentially harmful incidents like bullying or hate speech have predated theInternet, the reach and extent of Internet has given these incidents an unprecedented power and inﬂuenceto affect the lives of billions of people. It has been reported that these incidents have not only createdmental and psychological agony to the users of the web but has in fact forced people to deactivate theiraccounts and in extreme cases also commit suicides (Hinduja and Patchin, 2010). Thus the incidents ofaggression and unratiﬁed verbal behaviour have not remained a minor nuisance, but have acquired theform of a major criminal activity that affects a large number of people. It is therefore important thatpreventive measures can be taken to cope with abusive behaviour aggression online.One of the strategies to cope with aggressive behaviour online is to manually monitor and moderateuser-generated content, however, the amount and pace at which new data is being created on the web hasrendered manual methods of moderation and intervention almost completely impractical. As such theuse (semi-) automatic methods to identify such behaviour has become important and has attracted moreattention from the research community in recent years (Davidson et al., 2017; Malmasi and Zampieri,2017).This work is"
        ]
    }

test_data = {
    "user_input": example['question'],
    "response": example['answer'],
    "reference": example['ground_truth']
}

# A test sample containing user_input, response (the output from the LLM), and reference (the expected output from the LLM) as data points to evaluate the summary.

metric_bleu = BleuScore()
metric_rouge = RougeScore(rouge_type="rouge1", mode="recall")
test_data = SingleTurnSample(**test_data)
score_bleu = metric_bleu.single_turn_score(test_data)
score_rouge = metric_rouge.single_turn_score(test_data)

print(f"BLEU Score: {score_bleu:.4f}\n")
print(f"ROUGE Score: {score_rouge:.4f}\n")