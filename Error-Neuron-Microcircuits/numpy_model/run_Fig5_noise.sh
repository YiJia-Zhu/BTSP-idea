DIRECTORY="experiments/Fig5_rep_unit_noise"
for i in $(seq 0 50)
do
        python runner.py --params $DIRECTORY/runs/lr$i/params.json &  
done

DIRECTORY="experiments/Fig5_error_unit_noise"
for i in $(seq 0 40)
do
        python runner.py --params $DIRECTORY/runs/lr$i/params.json &  
done