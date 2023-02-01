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
RD = results/acdc_geo

# CFLAGS = -O
# DEBUG = --debug
BS = 8  # BS stands for Batch Size
K = 4  # K for class

G_RGX = (patient\d+_\d+_\d+)_\d+
NET = ENet
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]


# TRNFOLD = $(RD)/ce \
	$(RD)/ce_bl_geo_point_fast \
	$(RD)/ce_bl_int_point_fast \
	$(RD)/ce_bl_mbd_point \
	$(RD)/ce_bl_eucl_point_fast

TRNFOLD = ce ce_bl_geo_point_fast ce_bl_int_point_fast ce_bl_mbd_point ce_bl_eucl_point_fast 

TRN = $(addsuffix /testEval.csv, $(addprefix $(RD)/, $(TRNFOLD)))

all: $(TRN) view report



# Testing
$(RD)/%: data/ACDC-2D-GEO/test/img data/ACDC-2D-GEO/test/gt
	$(info $(green)$(CC) $(CFLAGS) main_test.py $@$(reset))
	$(CC) $(CFLAGS) main_test.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group \
		--workdir=$(dir $@) --csv=$(@F) --n_class=4 --metric_axis 1 2 3 \
		--grp_regex="$(G_RGX)" --network=$(NET) --ignore_norm_dataloader  $(DEBUG) \
		--folders="$(B_DATA)" --weights=$(dir $@)best.pkl --test_folder=test \
		




#iter000/test
#best_epoch/val

# Viewing
view: $(TRN)
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 3 --img_source data/ACDC-2D-GEO/test/img data/ACDC-2D-GEO/test/gt \
		$(addsuffix iter000/test, $(dir $^)) --crop 10 \
		--display_names gt $(notdir $(^D)) --no_contour -C $(K) --legend \
		--class_names Background "Right ventricle" Myocardium "Left ventricle"


report: $(TRN)
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(dir $(TRN)) --metrics test_3d_dsc test_dice --axises 1 2 3 \
		--detail_axises $(DEBUG)



