#!/bin/bash

shopt -s extglob


if [ "$#" -leq 1 ]; then
    echo "Specify two arguments.   ARG1: train, test or makegifs; ARG2: POEM or ACDC."
else 
    if [[ "$2" == "ACDC" ]]; then
        parentfold=results/acdc_geo_final/rep
        #acdc_geo/
        sub1=092_01_0_04
        sub2=014_01_0_00
        imname=patient
        dataset=acdc
    elif [[ "$2" == "POEM" ]]; then
        parentfold=results/poem_geo_final/rep
        #poem_geo/
        sub1=500018_0_098
        sub2=500403_0_096
        imname=subj
        dataset=poem
    else
        echo "Second argument should be ACDC or POEM!"; 
    fi
fi


if [[ "$1" == "train" ]]; then
    for rep in 1 2 3 4 5
    do
        make -f ${dataset}_geo.make REP=$rep #&
        #wait
    done
    #after all runs, you can do reporting and viewing?
    #make -f acdc_geo.make report  #can report handle multiple runs??
fi


if [[ "$1" == "test" ]]; then
    for rep in 1 2 3 4 5
    do
        make -f ${dataset}_test.make REP=$rep  
    done
fi


if [[ "$1" == "makegifs" ]]; then
    #make gifs
    
    for rep in 1 2 3 4 5
    do
        echo "Creating gifs for rep $rep !"
        for folder in $(ls -d $parentfold$rep/*/)
        do
            outname1="$parentfold$rep/$(basename $folder)_$sub1.gif"
            outname2="$parentfold$rep/$(basename $folder)_$sub2.gif"
            echo "Creating $(basename $outname1), $(basename $outname2)"
            convert "$folder""iter"*"/val/$imname$sub1.png" $outname1
            mogrify -normalize $outname1
            convert "$folder""iter"*"/val/$imname$sub2.png" $outname2
            mogrify -normalize $outname2
        done
    done
fi

