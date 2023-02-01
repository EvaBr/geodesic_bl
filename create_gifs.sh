#!/bin/bash

parentfold=results/acdc_geo/shift_crop
subject=012_01_0_02
for folder in $(ls -d $parentfold/*/)
do
    echo "$folder"
	outname="$parentfold/$(basename $folder)_$subject.gif"
    #echo "$outname"
	convert "$folder""iter"*"/val/patient$subject.png" $outname
	mogrify -normalize $outname
done