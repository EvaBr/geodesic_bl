#!/bin/bash

shopt -s extglob

#first do 5 reps training
if [[ "$1" == "train" ]]; then
    for rep in 1 2 3 4 5
    do
        #make -f acdc_geo.make REP=$rep #&
        make -f acdc_geo.make REP=$rep #&
        #wait

        #mv "results/acdc_geo_final/rep" "results/acdc_geo_final/rep$rep"
    done
    #after all runs, you can do reporting and viewing?
    #make -f acdc_geo.make report  #can report handle multiple runs??
fi

if [[ "$1" == "test" ]]; then
    for rep in 1 2 3 4 5
    do
        make -f acdc_test.make REP=$rep  
    done
fi


if [[ "$1" == "makegifs" ]]; then
    #make gifs
    parentfold=results/acdc_geo_final/rep
    #acdc_geo/
    sub1=092_01_0_04
    sub2=014_01_0_00
    for rep in 1 2 3 4 5
    do
        echo "Creating gifs for rep $rep !"
        for folder in $(ls -d $parentfold$rep/*/)
        do
            outname1="$parentfold$rep/$(basename $folder)_$sub1.gif"
            outname2="$parentfold$rep/$(basename $folder)_$sub2.gif"
            echo "Creating $(basename $outname1), $(basename $outname2)"
            convert "$folder""iter"*"/val/patient$sub1.png" $outname1
            mogrify -normalize $outname1
            convert "$folder""iter"*"/val/patient$sub2.png" $outname2
            mogrify -normalize $outname2
        done
    done
fi

if [[ "$1" == "poemgifs" ]]; then
    #make gifs
    parentfold=results/poem_geo_newDTs_nobckg
    #acdc_geo/
    sub1=500018_0_098
    sub2=500403_0_096
    for folder in $(ls -d $parentfold$rep/*/)
    do
        outname1="$parentfold$rep/$(basename $folder)_$sub1.gif"
        outname2="$parentfold$rep/$(basename $folder)_$sub2.gif"
        echo "Creating $(basename $outname1), $(basename $outname2)"
        convert "$folder""iter"*"/val/subj$sub1.png" $outname1
        mogrify -normalize $outname1
        convert "$folder""iter"*"/val/subj$sub2.png" $outname2
        mogrify -normalize $outname2
    done
fi
