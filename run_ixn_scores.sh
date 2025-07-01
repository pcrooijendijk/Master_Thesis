#!/bin/bash

for i in {1..11}
do
    echo "Processing dataset $i..."
    python3 test/ixn_score.py --path retrieved_docs_p/retrieved_docs_${i}.json --output test_ixn_${i}.pkl
done

echo "All done!"
