CC = python3.9
SHELL = /usr/bin/zsh
PP = PYTHONPATH="$(PYTHONPATH):."

# RD stands for Result DIR -- useful way to report from extracted archive
RD = minipaper/results/SYNT/

#.PHONY = all boundary plot train metrics hausdorff pack
.PHONY = train eval

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
reset:=$(shell tput sgr0)

# CFLAGS = -O
# DEBUG = --debug
EPC = 30
# EPC = 5

K = 2 #num classes
BS = 16
G_RGX = (case_\d+_\d+)_\d+
P_RGX = (case_\d+)_\d+_\d+
# NET = ResidualUNet
NET = UNet
B_DATA = [('IN', npy_transform, False), ('GT', gt_transform, True)] 
B_DATA_N = [('IN', npy_transform, False), ('GT_noisy', gt_transform, True), ('GT', gt_transform, True)] 

EVALDATA = val/GT
EVALCSV = eval.csv 
TARGETLIST = ['orig', 'orig_n', 'mbd', 'geo', 'ambd', 'ageo', 'euc']


TRN =  $(RD)/orig $(RD)/orig_n $(RD)/mbd $(RD)/geo $(RD)/ambd $(RD)/ageo $(RD)/euc 
EVAL = $(RD)/$(EVALCSV)

#GRAPH = $(RD)/tra_loss.png $(RD)/val_loss.png \
		$(RD)/val_dice.png $(RD)/tra_dice.png \

#PLT = $(GRAPH) 

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-poem.tar.gz

#all: $(PACK)

#plot: $(PLT)

train: $(TRN)

eval: $(EVAL)

#pack: report $(PACK)
#$(PACK): $(PLT) $(TRN)
#	$(info $(red)tar cf $@$(reset))
#	mkdir -p $(@D)
#	tar cf - $^ | tar -zc -f $@ $^ 

# tar -zc -f $@ $^  # Use if pigz is not available. othwise pigz > $@
#$(LIGHTPACK): $(PLT) $(TRN)
#	mkdir -p $(@D)
#	$(eval PLTS:=$(filter %.png, $^))
#	$(eval FF:=$(filter-out %.png, $^))
#	$(eval TGT:=$(addsuffix /best_epoch, $(FF)) $(addsuffix /*.npy, $(FF)) $(addsuffix /best_epoch.txt, $(FF)) $(addsuffix /metrics.csv, $(FF)))
#	tar cf - $(PLTS) $(TGT) | tar -zc -f $@ $^ 
#	chmod -w $@


# Training
$(RD)/orig: OPT = --losses="[('GeneralizedDice', {'idc': [0,1]}, 1)]"
$(RD)/orig: minipaper/data_synt/train/IN minipaper/data_synt/val/IN 
$(RD)/orig: DATA = --folders="$(B_DATA)+[('GT', gt_transform, True)]"

$(RD)/orig_n: OPT = --losses="[('GeneralizedDice', {'idc': [0,1]}, 1)]"
$(RD)/orig_n: minipaper/data_synt/train/IN minipaper/data_synt/val/IN 
$(RD)/orig_n: DATA = --folders="$(B_DATA_N)+[('GT_noisy', gt_transform, True)]"
$(RD)/orig_n: NOISY = --compute_on_pts


$(RD)/euc: OPT = --losses="[('SurfaceLoss', {'idc': [1]}, 1)]"
$(RD)/euc: minipaper/data_synt/train/IN minipaper/data_synt/val/IN 
$(RD)/euc: DATA = --folders="$(B_DATA_N)+[('GT_noisy', dist_map_transform, False)]" 
$(RD)/euc: NOISY = --compute_on_pts

$(RD)/geo: OPT = --losses="[('SurfaceLoss', {'idc': [1]}, 1)]"
$(RD)/geo: minipaper/data_synt/train/IN minipaper/data_synt/val/IN 
$(RD)/geo: DATA = --folders="$(B_DATA_N)+[('GTn_IN/GEO', tensorT_transform, False)]" 
$(RD)/geo: NOISY = --compute_on_pts

$(RD)/mbd: OPT = --losses="[('SurfaceLoss', {'idc': [1]}, 1)]"
$(RD)/mbd: minipaper/data_synt/train/IN minipaper/data_synt/val/IN 
$(RD)/mbd: DATA = --folders="$(B_DATA_N)+[('GTn_IN/MBD', tensorT_transform, False)]" 
$(RD)/mbd: NOISY = --compute_on_pts

$(RD)/ageo: OPT = --losses="[('SurfaceLoss', {'idc': [1]}, 1)]"
$(RD)/ageo: minipaper/data_synt/train/IN minipaper/data_synt/val/IN 
$(RD)/ageo: DATA = --folders="$(B_DATA_N)+[('aGTn_IN/GEO', tensorT_transform, False)]" 
$(RD)/ageo: NOISY = --compute_on_pts

$(RD)/ambd: OPT = --losses="[('SurfaceLoss', {'idc': [1]}, 1)]"
$(RD)/ambd: minipaper/data_synt/train/IN minipaper/data_synt/val/IN 
$(RD)/ambd: DATA = --folders="$(B_DATA_N)+[('aGTn_IN/MBD', tensorT_transform, False)]" 
$(RD)/ambd: NOISY = --compute_on_pts






$(RD)/%:
	$(info $(green)$(CC) $(CFLAGS) main.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	printenv > $@_tmp/env.txt
	git diff > $@_tmp/repo.diff
	git rev-parse --short HEAD > $@_tmp/commit_hash
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --in_memory --l_rate=0.001 --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=$(K) --modalities=1 --metric_axis 0 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(NOISY) $(OPT) $(DATA) $(DEBUG) 
	mv $@_tmp $@


$(RD)/$(EVALCSV):
	$(CC) $(CFLAGS) finalEval.py --dataset=$(EVALDATA) --csv=$(EVALCSV) \
		--savedir=$@ --folders=$(TARGETLIST) --nrepochs=$(EPC)
#--compute_3d_dice \

# Metrics
## Those need to be computed once the training is over, as we have to reconstruct the whole 3D volume


#hausdorff: $(TRN) \
	$(addsuffix /val_hausdorff.npy, $(TRN))


#boundary: $(TRN) \
	$(addsuffix /val_boundary.npy, $(TRN))


# Plotting
#$(RD)/tra_loss.png $(RD)/val_loss.png: COLS = 0 1
#$(RD)/tra_loss.png $(RD)/val_loss.png: OPT = --ylim -1 1 --dynamic_third_axis --no_mean
#$(RD)/tra_loss.png $(RD)/val_loss.png: plot.py $(TRN)

#$(RD)/val_dice.png $(RD)/tra_dice.png: COLS = 1
#$(RD)/val_dice.png $(RD)/tra_dice.png: plot.py $(TRN)



