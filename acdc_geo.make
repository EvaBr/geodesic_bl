CC = python3
PP = PYTHONPATH="$(PYTHONPATH):."
SHELL = zsh


.PHONY: mbddist mbddistnorm200 all euclid geodist geodistF intensity train plot view view_euclid view_labels view_superpixels npy pack report weak

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)

# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/acdc_geo/augmented

# CFLAGS = -O
# DEBUG = --debug
EPC = 100
BS = 8  # BS stands for Batch Size
K = 4  # K for class

G_RGX = (patient\d+_\d+_\d+)_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NET = ENet
# NET = Dummy


TRN = $(RD)/ce \
	$(RD)/ce_bl_geo_point_fast \
	$(RD)/ce_bl_int_point_fast \
	$(RD)/ce_bl_mbd_point \
	$(RD)/ce_bl_eucl_point_fast
#	$(RD)/ce_bl_eucl_point_fast_norm200 \
#	$(RD)/ce_bl_geo_point_fast_norm200 \
#	$(RD)/ce_bl_mbd_point_norm200 \
#	$(RD)/ce_bl_mbd_nn_point_smooth \
#	$(RD)/ce_bl_mbd_point_smooth \
# 	$(RD)/ce_bl_eucl_point_live \
# 	$(RD)/ce_bl_eucl_point_numpy \
#	$(RD)/ce_weak \
#	$(RD)/ce_bl_geo_point_fast2 \
#	$(RD)/ce_bl_mix_point_fast \
#	$(RD)/ce_bl_geo_point \
#   $(RD)/ce_bl_geo_nn_point \


GRAPH = $(RD)/val_dice.png $(RD)/tra_dice.png \
                $(RD)/tra_loss.png \
                $(RD)/val_3d_dsc.png
HIST =  $(RD)/val_dice_hist.png $(RD)/tra_loss_hist.png \
                $(RD)/val_3d_dsc_hist.png
BOXPLOT = $(RD)/val_3d_dsc_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-acdc_geo.tar.zst

all: pack

train: $(TRN)
plot: $(PLT)

pack: $(PACK) report view
$(PACK): $(PLT) $(TRN)
	$(info $(red)tar cf $@$(reset))
	mkdir -p $(@D)
	tar cf - $^ | zstd -T0 -3 > $@
	chmod -w $@
#       tar cf $@ - $^
#       tar cf - $^ | pigz > $@
#       tar cvf - $^ | zstd -T0 -#3 > $@
# tar -zc -f $@ $^  # Use if pigz is not available


# Data generation
data/ACDC-2D-GEO: OPT = --seed=0 --retains 10 --retains_test 25
data/ACDC-2D-GEO: data/acdc
	$(info $(yellow)$(CC) $(CFLAGS) preprocess/slice_acdc.py$(reset))
	rm -rf $@_tmp $@
	$(PP) $(CC) $(CFLAGS) preprocess/slice_acdc.py --source_dir="data/acdc/training" --dest_dir=$@_tmp $(OPT)
	mv $@_tmp $@

data/acdc: data/acdc.lineage data/acdc.zip
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	rm $@_tmp/training/*/*_4d.nii.gz  # space optimization
	mv $@_tmp $@

weaks = data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random
weak: $(weaks)

data/ACDC-2D-GEO/train/img data/ACDC-2D-GEO/val/img: | data/ACDC-2D-GEO
data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/val/gt: | data/ACDC-2D-GEO

data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat

$(weaks): data/ACDC-2D-GEO
	$(info $(yellow)$(CC) $(CFLAGS) gen_weak.py $@ $(reset))
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py -K $(K) --base_folder=$(@D) --GT_subfolder gt --save_subfolder=$(@F)_tmp --selected_class 1 2 3 --filling 1 2 3 $(OPT) --verbose
	mv $@_tmp $@


npys = data/ACDC-2D-GEO/train/gt_npy data/ACDC-2D-GEO/val/gt_npy \
	data/ACDC-2D-GEO/train/img_npy data/ACDC-2D-GEO/val/img_npy \
	data/ACDC-2D-GEO/train/random_npy data/ACDC-2D-GEO/val/random_npy
npy: $(npys)

data/ACDC-2D-GEO/train/gt_npy data/ACDC-2D-GEO/val/gt_npy: MODE = to_one_hot_npy
data/ACDC-2D-GEO/train/img_npy data/ACDC-2D-GEO/val/img_npy: MODE = to_npy
data/ACDC-2D-GEO/train/random_npy data/ACDC-2D-GEO/val/random_npy: MODE = to_npy

%_npy: %
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_npy $@ $(reset))
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) map_png.py --mode $(MODE) --src $< --dest $@_tmp -K $(K)
	mv $@_tmp $@


euclid: data/ACDC-2D-GEO/train/eucl_point_fast data/ACDC-2D-GEO/val/eucl_point_fast

data/ACDC-2D-GEO/%/eucl_point_fast: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode euclidean $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_distmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode euclidean
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3


geodist: data/ACDC-2D-GEO/train/geo_point_fast data/ACDC-2D-GEO/val/geo_point_fast
mbddist: data/ACDC-2D-GEO/train/mbd_point data/ACDC-2D-GEO/val/mbd_point 

data/ACDC-2D-GEO/%/geo_point_fast: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode geodesic $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_distmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode geodesic 
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3


data/ACDC-2D-GEO/%/mbd_point: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_dmap --distmap_mode intensity $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_dmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode intensity
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3
	

intensity: data/ACDC-2D-GEO/train/int_point_fast data/ACDC-2D-GEO/val/int_point_fast

data/ACDC-2D-GEO/%/int_point_fast: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode intensity $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_distmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode intensity
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3



superpixel: data/ACDC-2D-GEO/train/point_superpixel data/ACDC-2D-GEO/val/point_superpixel

data/ACDC-2D-GEO/%/point_superpixel: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode intensity $@ $(reset))
	rm -rf $@_tmp $@_tmp_raw
	mkdir -p $@_tmp $@_tmp_raw
	$(CC) $(CFLAGS) map_png.py --mode to_superpixel --src $< $(<D)/img --dest $@_tmp \
		-K $(K)
	mv $@_tmp $@
	mv $@_tmp_raw $@_raw


# Trainings
## Full ones
$(RD)/ce: OPT = --losses="[('CrossEntropy', {'idc': [0, 1, 2, 3]}, 1)]"
$(RD)/ce: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/val/gt
$(RD)/ce: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

## Weak ones
### Combined
#### Euclidean
$(RD)/ce_bl_eucl_point_fast: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/ce_bl_eucl_point_fast: data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random data/ACDC-2D-GEO/train/eucl_point_fast data/ACDC-2D-GEO/val/eucl_point_fast | npy euclid
$(RD)/ce_bl_eucl_point_fast: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('random_npy', gt_transform, True), ('eucl_point_fast', from_numpy_transform, False)]"


#### Geodesic
$(RD)/ce_bl_geo_point_fast: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/ce_bl_geo_point_fast: data/ACDC-2D-GEO/train/random_npy data/ACDC-2D-GEO/val/random_npy data/ACDC-2D-GEO/train/geo_point_fast data/ACDC-2D-GEO/val/geo_point_fast | npy geodist
$(RD)/ce_bl_geo_point_fast: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('random_npy', gt_transform, True), ('geo_point_fast', from_numpy_transform, False)]"


#### Intensity
$(RD)/ce_bl_int_point_fast: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/ce_bl_int_point_fast: data/ACDC-2D-GEO/train/random_npy data/ACDC-2D-GEO/val/random_npy data/ACDC-2D-GEO/train/int_point_fast data/ACDC-2D-GEO/val/int_point_fast | npy intensity
$(RD)/ce_bl_int_point_fast: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('random_npy', gt_transform, True), ('int_point_fast', from_numpy_transform, False)]"


#### MBD
$(RD)/ce_bl_mbd_point: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/ce_bl_mbd_point: data/ACDC-2D-GEO/train/random_npy data/ACDC-2D-GEO/val/random_npy data/ACDC-2D-GEO/train/mbd_point data/ACDC-2D-GEO/val/mbd_point | npy mbddist
$(RD)/ce_bl_mbd_point: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('random_npy', gt_transform, True), ('mbd_point', from_numpy_transform, False)]"





# Template
$(RD)/%:
	$(info $(green)$(CC) $(CFLAGS) main.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	printenv > $@_tmp/env.txt
	git diff > $@_tmp/repo.diff
	git rev-parse --short HEAD > $@_tmp/commit_hash
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=4 --metric_axis 1 2 3 \
		--compute_3d_dice --augment \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Plotting
$(RD)/val_3d_dsc.png $(RD)/val_dice.png $(RD)/tra_dice.png: COLS = 1 2 3
$(RD)/tra_loss.png: COLS = 0
$(RD)/tra_loss.png: OPT = --dynamic_third_axis
$(RD)/val_dice.png $(RD)/tra_loss.png $(RD)/val_3d_dsc.png: plot.py $(TRN)
$(RD)/tra_dice.png: plot.py $(TRN)

$(RD)/val_rate.png $(RD)/tra_rate.png: plot.py $(TRN)
$(RD)/val_rate.png $(RD)/tra_rate.png: OPT = --ylim 0 25
$(RD)/val_rate.png $(RD)/tra_rate.png: COLS = 0

$(RD)/tra_ClECE.png $(RD)/val_ClECE.png: COLS = 0 2 3 4
$(RD)/tra_ClECE.png $(RD)/val_ClECE.png: OPT = --ylim 0 0.1
$(RD)/tra_ClECE.png $(RD)/val_ClECE.png: plot.py $(TRN)

$(RD)/val_3d_dsc_hist.png $(RD)/val_dice_hist.png: COLS = 3
$(RD)/tra_loss_hist.png: COLS = 0
$(RD)/tra_loss_hist.png: OPT = --dynamic_third_axis
$(RD)/val_dice_hist.png $(RD)/val_3d_dsc_hist.png $(RD)/tra_loss_hist.png: hist.py $(TRN)

$(RD)/val_3d_dsc_boxplot.png: COLS = 1 2 3
$(RD)/val_3d_dsc_boxplot.png: moustache.py $(TRN)

$(RD)/%.png:
	$(info $(blue)$(CC) $(CFLAGS) $< $@$(reset))
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT)



# Viewing
view: $(TRN) | weak
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/val/img data/ACDC-2D-GEO/val/gt \
		$(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^) --no_contour -C $(K) --legend \
		--class_names Background "Right ventricle" Myocardium "Left ventricle"

view_euclid: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/train/eucl_point_numpy_1 data/ACDC-2D-GEO/train/eucl_point_numpy_2 data/ACDC-2D-GEO/train/eucl_point_numpy_3 data/ACDC-2D-GEO/train/eucl_point_fast_1 data/ACDC-2D-GEO/train/eucl_point_fast_2 data/ACDC-2D-GEO/train/eucl_point_fast_3
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/train/img $^ \
		--display_names $(notdir $^) --no_contour --cmap viridis --alpha 1.0 --legend -C 256

view_labels: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/train/random \
	data/ACDC-2D-GEO/train/eucl_point_fast_1 data/ACDC-2D-GEO/train/eucl_point_fast_2 data/ACDC-2D-GEO/train/eucl_point_fast_3 \
	data/ACDC-2D-GEO/train/geo_point_fast_1 data/ACDC-2D-GEO/train/geo_point_fast_2 data/ACDC-2D-GEO/train/geo_point_fast_3 \
	data/ACDC-2D-GEO/train/int_point_fast_1 data/ACDC-2D-GEO/train/int_point_fast_2 data/ACDC-2D-GEO/train/int_point_fast_3
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/train/img $^ \
		--display_names $(notdir $^) --no_contour --cmap viridis --alpha 1.0 --legend -C 256


view_superpixels: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/train/point_superpixel data/ACDC-2D-GEO/train/point_superpixel_raw
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/train/img $^ \
		--display_names $(notdir $^) -C 150
# 		--display_names $(notdir $^) --no_contour --alpha 0.8 --legend -C 256

report: $(TRN)
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_3d_dsc val_dice --axises 1 2 3 \
		--detail_axises $(DEBUG)
