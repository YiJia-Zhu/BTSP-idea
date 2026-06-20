DIRECTORY="experiments/Fig4_populations"
for i in $(seq 0 160)
do
        python runner.py --params $DIRECTORY/runs/lr$i/params.json &  
done
