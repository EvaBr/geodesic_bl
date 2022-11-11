CC = python3.10
PP = PYTHONPATH="$(PYTHONPATH):."
SHELL = zsh


.PHONY: all euclid geodist train plot view view_euclid view_labels npy pack report weak

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)

# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/acdc_geo

# CFLAGS = -O
# DEBUG = --debug
EPC = 300
BS = 8  # BS stands for Batch Size
K = 4  # K for class

G_RGX = (patient\d+_\d+_\d+)_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NET = ENet
# NET = Dummy


TRN = $(RD)/ce \
        $(RD)/ce_boundary_geo_point \
        $(RD)/ce_boundary_geo_point_master \
	$(RD)/ce_boundary_eucl_point \
	$(RD)/ce_boundary_int_point \


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

pack: $(PACK) report
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


geodist: data/ACDC-2D-GEO/train/geodist data/ACDC-2D-GEO/val/geodist \
	data/ACDC-2D-GEO/train/geodist_point data/ACDC-2D-GEO/val/geodist_point \
	data/ACDC-2D-GEO/train/intensity_point data/ACDC-2D-GEO/val/intensity_point
euclid: data/ACDC-2D-GEO/train/euclid_point data/ACDC-2D-GEO/val/euclid_point

data/ACDC-2D-GEO/%/geodist: data/ACDC-2D-GEO/%/gt
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode geodesic $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_distmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode geodesic --norm_dist
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3
data/ACDC-2D-GEO/%/geodist_point: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode geodesic $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_distmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode geodesic --norm_dist
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3
data/ACDC-2D-GEO/%/euclid_point: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode euclidean $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_distmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode euclidean --norm_dist
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3
data/ACDC-2D-GEO/%/intensity_point: data/ACDC-2D-GEO/%/random
	$(info $(yellow)$(CC) $(CFLAGS) map_png.py --mode to_distmap --distmap_mode intensity $@ $(reset))
	rm -rf $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	mkdir -p $@_tmp $@_tmp_0 $@_tmp_1 $@_tmp_2 $@_tmp_3
	$(CC) $(CFLAGS) map_png.py --mode to_distmap --src $< $(<D)/img --dest $@_tmp \
		-K $(K) --distmap_mode intensity --norm_dist
	mv $@_tmp $@
	mv $@_tmp_0 $@_0
	mv $@_tmp_1 $@_1
	mv $@_tmp_2 $@_2
	mv $@_tmp_3 $@_3
#       rename -v '_tmp' '' $@_tmp*  # Doesn't work the same between distribs


# Trainings
## Full ones
$(RD)/ce: OPT = --losses="[('CrossEntropy', {'idc': [0, 1, 2, 3]}, 1)]"
$(RD)/ce: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/val/gt
$(RD)/ce: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

$(RD)/diceloss: OPT = --losses="[('DiceLoss', {'idc': [0, 1, 2, 3]}, 1)]"
$(RD)/diceloss: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/val/gt
$(RD)/diceloss: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

$(RD)/boundary_eucl: OPT = --losses="[('BoundaryLoss', {'idc': [0, 1, 2, 3]}, 1)]"
$(RD)/boundary_eucl: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/val/gt
$(RD)/boundary_eucl: DATA = --folders="$(B_DATA)+[('gt', dist_map_transform, False)]"

$(RD)/boundary_geo: OPT = --losses="[('BoundaryLoss', {'idc': [0, 1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/boundary_geo: data/ACDC-2D-GEO/train/geodist data/ACDC-2D-GEO/val/geodist | npy geodist
$(RD)/boundary_geo: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('geodist', from_numpy_transform, False)]"


## Weak ones
### Single loss
$(RD)/ce_point: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1)]"
$(RD)/ce_point: data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random
$(RD)/ce_point: DATA = --folders="$(B_DATA)+[('random', gt_transform, True)]"

$(RD)/diceloss_point: OPT = --losses="[('DiceLoss', {'idc': [1, 2, 3]}, 1)]"
$(RD)/diceloss_point: data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random
$(RD)/diceloss_point: DATA = --folders="$(B_DATA)+[('random', gt_transform, True)]"

$(RD)/boundary_eucl_point: OPT = --losses="[('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]"
$(RD)/boundary_eucl_point: data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random
$(RD)/boundary_eucl_point: DATA = --folders="$(B_DATA)+[('random', dist_map_transform, False)]"

$(RD)/boundary_geo_point: OPT = --losses="[('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/boundary_geo_point: data/ACDC-2D-GEO/train/geodist_point data/ACDC-2D-GEO/val/geodist_point | npy geodist
$(RD)/boundary_geo_point: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('geodist_point', from_numpy_transform, False)]"


### Combined
# $(RD)/ce_boundary_eucl_point: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
# 	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
# $(RD)/ce_boundary_eucl_point: data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random data/ACDC-2D-GEO/train/euclid_point data/ACDC-2D-GEO/val/euclid_point | npy euclid
# $(RD)/ce_boundary_eucl_point: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('random_npy', gt_transform, True), ('euclid_point', from_numpy_transform, False)]"

$(RD)/ce_boundary_eucl_point: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]"
$(RD)/ce_boundary_eucl_point: data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/val/random
$(RD)/ce_boundary_eucl_point: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', dist_map_transform, False)]"

$(RD)/ce_boundary_geo_point: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/ce_boundary_geo_point: data/ACDC-2D-GEO/train/random_npy data/ACDC-2D-GEO/val/random_npy data/ACDC-2D-GEO/train/geodist_point data/ACDC-2D-GEO/val/geodist_point | npy geodist
$(RD)/ce_boundary_geo_point: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('random_npy', gt_transform, True), ('geodist_point', from_numpy_transform, False)]"


$(RD)/ce_boundary_int_point: OPT = --losses="[('CrossEntropy', {'idc': [1, 2, 3]}, 1),\
	('BoundaryLoss', {'idc': [1, 2, 3]}, 1)]" --ignore_norm_dataloader
$(RD)/ce_boundary_int_point: data/ACDC-2D-GEO/train/random_npy data/ACDC-2D-GEO/val/random_npy data/ACDC-2D-GEO/train/intensity_point data/ACDC-2D-GEO/val/intensity_point | npy geodist
$(RD)/ce_boundary_int_point: DATA = --folders="[('img_npy', npy_transform, False), ('gt_npy', from_numpy_transform, True), ('random_npy', gt_transform, True), ('intensity_point', from_numpy_transform, False)]"


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
		--compute_3d_dice \
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

view_euclid: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/train/random data/ACDC-2D-GEO/train/euclid_point_1 data/ACDC-2D-GEO/train/euclid_point_2 data/ACDC-2D-GEO/train/euclid_point_3
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/train/img $^ \
		--display_names $(notdir $^) --no_contour --cmap viridis --alpha 1.0 --legend -C $(K)

# view_labels: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/train/random
view_labels: data/ACDC-2D-GEO/train/gt data/ACDC-2D-GEO/train/random \
	data/ACDC-2D-GEO/train/intensity_point_1 data/ACDC-2D-GEO/train/intensity_point_2 data/ACDC-2D-GEO/train/intensity_point_3 \
	data/ACDC-2D-GEO/train/geodist_point_1 data/ACDC-2D-GEO/train/geodist_point_2 data/ACDC-2D-GEO/train/geodist_point_3 \
	data/ACDC-2D-GEO/train/euclid_point_1 data/ACDC-2D-GEO/train/euclid_point_2 data/ACDC-2D-GEO/train/euclid_point_3
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/train/img $^ \
		--display_names $(notdir $^) --no_contour --cmap viridis --alpha 1.0 --legend -C $(K)

report: $(TRN)
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_3d_dsc val_dice --axises 1 2 3 \
		--detail_axises $(DEBUG)
