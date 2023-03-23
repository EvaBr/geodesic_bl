CC = python3
PP = PYTHONPATH="$(PYTHONPATH):."
SHELL = zsh


.PHONY: all view report

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)

# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/acdc_geo_final_partialdata/rep$(REP)
# CFLAGS = -O
# DEBUG = --debug
BS = 1  # BS stands for Batch Size
K = 4  # K for class

G_RGX = (patient\d+_\d+_\d+)_\d+
NET = ENet
#B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]


# TRNFOLD = $(RD)/ce \
	$(RD)/ce_bl_geo_point_fast \
	$(RD)/ce_bl_int_point_fast \
	$(RD)/ce_bl_mbd_point \
	$(RD)/ce_bl_eucl_point_fast

TRNFOLD = ce ce_weak ce_bl_geo_point_fast ce_bl_int_point_fast ce_bl_mbd_point ce_bl_eucl_point_fast 
TRNS = $(addprefix $(RD)/, $(TRNFOLD))

#TRN = $(addsuffix /testEval.csv, $(addprefix $(RD)/, $(TRNFOLD)))
TRN = $(addsuffix /iter000/test, $(TRNS))


all: $(TRN) metrics report #view 
#all: view


#$(RD)/ce_norm/testEval.csv: B_DATA=[('img', png_transform, False), ('gt', gt_transform, True)]
#$(RD)/ce_bl%/testEval.csv $(RD)/ce/testEval.csv: B_DATA=[('img', png_transform_nonorm, False), ('gt', gt_transform, True)]

# Testing
#$(RD)/%: data/ACDC-2D-GEO/test/img data/ACDC-2D-GEO/test/gt
#	$(info $(green)$(CC) $(CFLAGS) main_test.py $@$(reset))
#	$(CC) $(CFLAGS) main_test.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group \
		--workdir=$(dir $@) --csv=$(@F) --n_class=4 --metric_axis 1 2 3 \
		--grp_regex="$(G_RGX)" --network=$(NET) --ignore_norm_dataloader  $(DEBUG) \
		--folders="$(B_DATA)" --weights=$(dir $@)best.pkl --test_folder=test \


$(RD)/%: data/ACDC-2D-GEO/test/img data/ACDC-2D-GEO/test/gt
	$(info $(green)$(CC) $(CFLAGS) inference.py $@$(reset))
	$(CC) $(CFLAGS) inference.py --data_folder=$(<) --batch_size=$(BS) \
		--num_classes=4 --group $(DEBUG) --csv=metrics_test.csv --grp_regex="$(G_RGX)" \
		--save_folder=$(dir $(@D)) --model_weights=$(dir $(@D))best.pkl \
       

metrics: $(addsuffix /test_3d_dsc.npy, $(TRNS)) \
	$(addsuffix /test_3d_hd95.npy, $(TRNS))
#	$(TRN) \
#	$(addsuffix /test_3d_hausdorff.npy, $(TRNS)) \


$(RD)/%/test_3d_dsc.npy $(RD)/%/test_3d_hausdorff.npy $(RD)/%/test_3d_hd95.npy: data/ACDC-2D-GEO/test/gt | $(RD)/%
	$(info $(green)$(CC) $(CFLAGS) metrics_overtime.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics 3d_dsc 3d_hd95 \
		--grp_regex "$(G_RGX)" \
		--num_classes $(K) --n_epoch 1 --mode=test \
		--gt_folder $^

#iter000/test
#best_epoch/val

# Viewing
#view: $(TRN)
#	$(info $(cyan)$(CC) viewer/viewer.py  $(dir $^) $(reset))
#	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/test/img data/ACDC-2D-GEO/test/gt \
		$(addsuffix iter000/test, $(dir $^))  --crop 10 \
		--display_names gt $(notdir $(^D)) --no_contour -C $(K) --legend \
		--class_names Background "Right ventricle" Myocardium "Left ventricle"
view: $(TRN)
	$(info $(cyan)$(CC) viewer/viewer.py  $(TRNS) $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/test/img data/ACDC-2D-GEO/test/gt $^ \
		--crop 10 \
		--display_names gt $(TRNFOLD) --no_contour -C $(K) --legend \
		--class_names Background "Right ventricle" Myocardium "Left ventricle"


report: $(TRN)
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRNS) --metrics test_3d_dsc test_3d_hd95 --axises 1 2 3 \
		--detail_axises $(DEBUG)



