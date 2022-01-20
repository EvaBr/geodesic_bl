CC = python3.9
SHELL = /usr/bin/zsh
PP = PYTHONPATH="$(PYTHONPATH):."

# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/poem #BaselineOrig

.PHONY = all boundary plot train metrics hausdorff pack

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
reset:=$(shell tput sgr0)

# CFLAGS = -O
# DEBUG = --debug
EPC = 150
# EPC = 5

K = 7 #num classes
BS = 8
G_RGX = (case_\d+_\d+)_\d+
P_RGX = (case_\d+)_\d+_\d+
NET = ResidualUNet
B_DATA = [('in_npy', tensor_transform, False), ('gt_orig_npy', gt_transform, True)] #, ('gt_pts_npy', gt_transform, True)] #<- use gt_pts only together with flag --compute_on_pts!!!

TRN =  $(RD)/gdl_1 $(RD)/gdl_c $(RD)/gdl_w_1 $(RD)/gdl_w_c

GRAPH = $(RD)/tra_loss.png $(RD)/val_loss.png \
		$(RD)/val_dice.png $(RD)/tra_dice.png \
		$(RD)/val_3d_hausdorff.png \
		$(RD)/val_3d_hd95.png
BOXPLOT =  $(RD)/val_dice_boxplot.png
PLT = $(GRAPH) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-poem.tar.gz

all: $(PACK)

plot: $(PLT)

train: $(TRN)

pack: report $(PACK)
$(PACK): $(PLT) $(TRN)
	$(info $(red)tar cf $@$(reset))
	mkdir -p $(@D)
	tar cf - $^ | tar -zc -f $@ $^ 

# tar -zc -f $@ $^  # Use if pigz is not available. othwise pigz > $@
$(LIGHTPACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	$(eval PLTS:=$(filter %.png, $^))
	$(eval FF:=$(filter-out %.png, $^))
	$(eval TGT:=$(addsuffix /best_epoch, $(FF)) $(addsuffix /*.npy, $(FF)) $(addsuffix /best_epoch.txt, $(FF)) $(addsuffix /metrics.csv, $(FF)))
	tar cf - $(PLTS) $(TGT) | tar -zc -f $@ $^ 
	chmod -w $@


# Training
$(RD)/gdl_w_c: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.15, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
	('WeightedCrossEntropy', {'idc': [0.15,1,1,1,1,1,1]}, 1)]"
$(RD)/gdl_w_c: data/POEM/train/in_npy data/POEM/val/in_npy
$(RD)/gdl_w_c: DATA = --folders="$(B_DATA)+[('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True)]" 

$(RD)/gdl_w_1: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [1, 1, 1, 1, 1, 1, 1]}, 1), \
	('WeightedCrossEntropy', {'idc': [0.15,1,1,1,1,1,1]}, 1)]"
$(RD)/gdl_w_1: data/POEM/train/in_npy data/POEM/val/in_npy
$(RD)/gdl_w_1: DATA = --folders="$(B_DATA)+[('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True)]" 

$(RD)/gdl_c: OPT = --losses="[('GeneralizedDice', {'idc': [0]}, 0.15), \
	('GeneralizedDice', {'idc': [4]}, 0.5), \
	('GeneralizedDice', {'idc': [2]}, 0.4), \
	('GeneralizedDice', {'idc': [3]}, 0.35), \
	('GeneralizedDice', {'idc': [4]}, 0.5), \
	('GeneralizedDice', {'idc': [5]}, 0.4), \
	('GeneralizedDice', {'idc': [6]}, 0.4), \
	('CrossEntropy', {'idc': [0]}, 0.15), \
	('CrossEntropy', {'idc': [1,2,3,4,5,6]}, 1)]"
$(RD)/gdl_c: data/POEM/train/in_npy data/POEM/val/in_npy
$(RD)/gdl_c: DATA = --folders="$(B_DATA)+[('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True)]"

$(RD)/gdl_1: OPT = --losses="[('GeneralizedDice', {'idc': [0]}, 1), \
	('GeneralizedDice', {'idc': [1]}, 1), \
	('GeneralizedDice', {'idc': [2]}, 1), \
	('GeneralizedDice', {'idc': [3]}, 1), \
	('GeneralizedDice', {'idc': [4]}, 1), \
	('GeneralizedDice', {'idc': [5]}, 1), \
	('GeneralizedDice', {'idc': [6]}, 1), \
	('CrossEntropy', {'idc': [0]}, 0.15), \
	('CrossEntropy', {'idc': [1,2,3,4,5,6]}, 1)]"
$(RD)/gdl_1: data/POEM/train/in_npy data/POEM/val/in_npy
$(RD)/gdl_1: DATA = --folders="$(B_DATA)+[('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True), \
	('gt_orig_npy', gt_transform, True), ('gt_orig_npy', gt_transform, True)]"







$(RD)/%:
	$(info $(green)$(CC) $(CFLAGS) main.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	printenv > $@_tmp/env.txt
	git diff > $@_tmp/repo.diff
	git rev-parse --short HEAD > $@_tmp/commit_hash
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --in_memory --l_rate=0.001 --schedule \
		--use_spacing \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=7 --modalities=2 --metric_axis 0 1 2 3 4 5 6 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@
#--compute_3d_dice \

# Metrics
## Those need to be computed once the training is over, as we have to reconstruct the whole 3D volume
metrics: $(TRN) \
	$(addsuffix /val_3d_dsc.npy, $(TRN)) \
	$(addsuffix /val_3d_hausdorff.npy, $(TRN)) \
	$(addsuffix /val_3d_hd95.npy, $(TRN))

$(RD)/%/val_3d_dsc.npy $(RD)/%/val_3d_hausdorff.npy $(RD)/%/val_3d_hd95.npy: data/POEM/val/gt | $(RD)/%
	$(info $(green)$(CC) $(CFLAGS) metrics_overtime.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics 3d_dsc 3d_hausdorff 3d_hd95 \
		--grp_regex "$(G_RGX)" --resolution_regex "$(P_RGX)" \
		--spacing $(<D)/../spacing_3d.pkl \
		--num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $^


hausdorff: $(TRN) \
	$(addsuffix /val_hausdorff.npy, $(TRN))

$(RD)/%/val_hausdorff.npy: data/POEM/val/gt | $(RD)/%
	$(info $(green)$(CC) $(CFLAGS) metrics_overtime.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics hausdorff \
		--grp_regex "$(G_RGX)" --resolution_regex "$(P_RGX)" \
		--spacing $(<D)/../spacing_3d.pkl \
		--num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $^


boundary: $(TRN) \
	$(addsuffix /val_boundary.npy, $(TRN))

$(RD)/%/val_boundary.npy: data/POEM/val/gt | $(RD)/%
	$(info $(green)$(CC) $(CFLAGS) metrics_overtime.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics boundary \
		--grp_regex "$(G_RGX)" --resolution_regex "$(P_RGX)" \
		--spacing $(<D)/../spacing_3d.pkl \
		--num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $^


# Plotting
$(RD)/tra_loss.png $(RD)/val_loss.png: COLS = 0 1
$(RD)/tra_loss.png $(RD)/val_loss.png: OPT = --ylim -1 1 --dynamic_third_axis --no_mean
$(RD)/tra_loss.png $(RD)/val_loss.png: plot.py $(TRN)

$(RD)/val_dice.png $(RD)/tra_dice.png: COLS = 1
$(RD)/val_dice.png $(RD)/tra_dice.png: plot.py $(TRN)

$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: COLS = 1
$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: OPT = --ylim 0 40 --min
$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: plot.py $(TRN)

$(RD)/val_dice_boxplot.png: COLS = 1
$(RD)/val_dice_boxplot.png: moustache.py $(TRN)

$(RD)/%.png: | metrics
	$(info $(blue)$(CC) $(CFLAGS) $< $@$(reset))
	$(eval metric:=$(subst _boxplot,,$(@F)))  # Needed to use same recipe for both histogram and plots
	$(eval metric:=$(subst _hist,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless $(OPT) $(DEBUG)


# Viewing
view: $(TRN)
	viewer/viewer.py -n 3 --img_source data/POEM/val/in_npy data/POEM/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(shell basename -a -s '/' $^) $(DEBUG)
	
report: $(TRN) | metrics
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRN) --axises 0 1 2 3 4 5 6 --precision 3 \
		--metrics val_dice val_hausdorff val_3d_dsc val_3d_hausdorff val_3d_hd95
