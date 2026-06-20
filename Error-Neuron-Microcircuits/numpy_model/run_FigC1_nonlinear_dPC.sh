DIRECTORY="experiments/FigC1_dPC_linear"
for net in 2-1 4-2-1 8-4-2-1 16-8-4-2-1 32-16-8-4-2-1
do
        for model in dPC
        do
                python runner.py --params $DIRECTORY/$net/$model/params.json --task fw_only --compare BP &  
                python runner.py --params $DIRECTORY/$net/untrained_$model/params.json --task fw_only --compare BP &  
        done
done

DIRECTORY="experiments/FigC1_dPC_nonlin_noUS"
for net in 2-1 4-2-1 8-4-2-1 16-8-4-2-1 32-16-8-4-2-1
do
        for model in dPC
        do
                python runner.py --params $DIRECTORY/$net/$model/params.json --task fw_only --compare BP &  
                python runner.py --params $DIRECTORY/$net/untrained_$model/params.json --task fw_only --compare BP &  
        done
done

DIRECTORY="experiments/FigC1_dPC_nonlin_US"
for net in 2-1 4-2-1 8-4-2-1 16-8-4-2-1 32-16-8-4-2-1
do
        for model in dPC
        do
                python runner.py --params $DIRECTORY/$net/$model/params.json --task fw_only --compare BP &  
                python runner.py --params $DIRECTORY/$net/untrained_$model/params.json --task fw_only --compare BP &  
        done
done