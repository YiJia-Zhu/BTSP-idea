DIRECTORY="experiments/FigA1_multilayer_comparison_hierarchical"
for net in 2-1 4-2-1 8-4-2-1 16-8-4-2-1 32-16-8-4-2-1
do
        for model in ann errormc sacramento2018 dPC
        do
                python runner.py --params $DIRECTORY/$net/$model/params.json --task fw_only --compare BP &  
                python runner.py --params $DIRECTORY/$net/untrained_$model/params.json --task fw_only --compare BP &  
        done
done