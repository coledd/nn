include ../../common.mk

.PHONY: all clean distclean stm32

DATASET_DIR := dataset
# Ninapro (https://ninapro.hevs.ch): DB2 is 40 intact subjects, DB3 11
# transradial amputees, both recorded with the same 12-electrode Delsys
# setup at 2kHz. Each subject is one zip of three exercises; only
# exercise 1 (basic hand and wrist movements) is used, so fetch_ninapro.py
# downloads just that file out of each zip with HTTP range requests:
# ~7GB for all 51 subjects rather than ~22GB.
NINAPRO_URL := https://ninapro.hevs.ch/files
DB2_FILES := $(foreach s,$(shell seq 1 40),$(DATASET_DIR)/db2_s$(s)_e1.mat)
DB3_FILES := $(foreach s,$(shell seq 1 11),$(DATASET_DIR)/db3_s$(s)_e1.mat)

# Electrodes used: the forearm ring (8) by default; 10 adds Ninapro's two
# targeted forearm electrodes, 12 the biceps/triceps pair as well. Every
# object and features.csv depend on a stamp file named for the setting, so
# changing it rebuilds everything that depends on it.
CHANNELS ?= 8
CFLAGS += -DEF_CHANNELS=$(CHANNELS)
CHANNELS_STAMP := .channels-$(CHANNELS)

all: train test replay features.csv

$(CHANNELS_STAMP):
	$(RM) .channels-*
	touch $@

# Recursing into the top-level Makefile (rather than duplicating nn.o's
# build rules here) keeps it compiled in exactly one place regardless of
# how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

# The on-device half: what ships to the prosthesis alongside nn.c.
DEVICE_OBJS := emg_features.o intent_decoder.o
# The desktop-only half: reading Ninapro's MATLAB files.
DESKTOP_OBJS := mat_reader.o ninapro.o

emg_features.o: emg_features.c emg_features.h $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -c $<

intent_decoder.o: intent_decoder.c intent_decoder.h emg_features.h ../../nn.h $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -I../.. -c $<

# mat_reader.c uses the zlib decoder inside ../../stb_image.h (vendored at
# the repository root, shared with image_classification and
# object_detection) to undo MATLAB's compression.
mat_reader.o: mat_reader.c mat_reader.h ../../stb_image.h $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -I../.. -c $<

ninapro.o: ninapro.c ninapro.h mat_reader.h intent_decoder.h emg_features.h $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -I../.. -c $<

prepare_data replay: %: %.c $(DESKTOP_OBJS) $(DEVICE_OBJS) ../../libnn.a $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -I../.. $< $(DESKTOP_OBJS) $(DEVICE_OBJS) ../../libnn.a $(LDFLAGS) -o $@

train test: %: %.c feature_set.h ninapro.h feature_set.o lda.o $(DEVICE_OBJS) ../../libnn.a $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -I../.. $< feature_set.o lda.o $(DEVICE_OBJS) ../../libnn.a $(LDFLAGS) -o $@

feature_set.o: feature_set.c feature_set.h intent_decoder.h emg_features.h $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -I../.. -c $<

lda.o: lda.c lda.h $(CHANNELS_STAMP)
	$(CC) $(CFLAGS) -c $<

# fetch_ninapro.py is an order-only prerequisite: editing or re-checking-out
# the script mustn't make 6GB of already-downloaded recordings out of date.
$(DATASET_DIR)/db2_s%_e1.mat: | fetch_ninapro.py
	mkdir -p $(DATASET_DIR)
	python fetch_ninapro.py $(NINAPRO_URL)/DB2_Preproc/DB2_s$*.zip /S$*_E1_A1.mat $@

$(DATASET_DIR)/db3_s%_e1.mat: | fetch_ninapro.py
	mkdir -p $(DATASET_DIR)
	python fetch_ninapro.py $(NINAPRO_URL)/db3_Preproc/s$*_0.zip /S$*_E1_A1.mat $@

features.csv: prepare_data $(DB2_FILES) $(DB3_FILES) $(CHANNELS_STAMP)
	./prepare_data $(DATASET_DIR) $@

# Cross-compiles the on-device half (plus nn.c) for the STM32U585
# (Cortex-M33, single-precision FPU) with the GNU Arm Embedded toolchain
# and reports how much flash and RAM it takes -- a check that it really is
# firmware-ready, not a substitute for a board bring-up. Needs only the
# toolchain (STM32CubeIDE bundles one; point ARM_CC at it if it isn't on
# PATH), not the HAL.
ARM_CC ?= arm-none-eabi-gcc
ARM_SIZE ?= $(patsubst %gcc,%size,$(ARM_CC))
ARM_CFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard -Os -ffunction-sections -fdata-sections -Wall -I../.. -DEF_CHANNELS=$(CHANNELS)
ARM_SOURCES := emg_features.c intent_decoder.c ../../nn.c

stm32:
	mkdir -p stm32
	for f in $(ARM_SOURCES); do $(ARM_CC) $(ARM_CFLAGS) -c $$f -o stm32/$$(basename $${f%.c}).o || exit 1; done
	$(ARM_SIZE) -t stm32/*.o

clean:
	$(RM) $(DEVICE_OBJS) $(DESKTOP_OBJS) feature_set.o lda.o prepare_data train test replay model*.txt model*.bin model*.inplace
	$(RM) -r stm32 .channels-*

distclean: clean
	$(RM) features.csv
	$(RM) -r $(DATASET_DIR)
