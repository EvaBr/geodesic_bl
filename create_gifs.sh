#!/bin/bash


parentfold=results/poem_loss_compare
#acdc_geo/
subject=500158_0_092
#092_01_0_04
for folder in $(ls -d $parentfold/*/)
do
	outname="$parentfold/$(basename $folder)_$subject.gif"
	echo "Creating $outname"
	convert "$folder""iter"*"/val/subj$subject.png" $outname
	mogrify -normalize $outname
done

#convert "$folder""iter"*"/val/patient$subject.png" $outname
	
