#!/bin/bash

for i in {1..11}
do
    echo "Processing dataset $i..."
    python3 test/RAGAS_bleu.py --path eval_dataset_p/eval_dataset_${i}.json --output BLEU_scores/scores_${i}.csv
done

echo "All done!"
