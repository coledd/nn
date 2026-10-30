include ../../common.mk

CSV_OUTPUTS := test.csv train.csv validation.csv
DATASET_DIR := dataset
DATASET_ARCHIVE := speech_commands_v0.02.tar.gz
DATASET_URL := https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
WAKE_WORD := marvin

.PHONY: all clean distclean

all: train_gru train_cnn test predict $(CSV_OUTPUTS)

# Recursing into the top-level Makefile (rather than duplicating nn.o's/
# data_prep.o's/audio_features.o's/fft.o's/wav_reader.o's build rules here) keeps
# each compiled in exactly one place regardless of how many examples/*/
# directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

../../data_prep.o:
	$(MAKE) -C ../.. data_prep.o

../../audio_features.o:
	$(MAKE) -C ../.. audio_features.o

../../fft.o:
	$(MAKE) -C ../.. fft.o

../../wav_reader.o:
	$(MAKE) -C ../.. wav_reader.o

train_gru: train_gru.c ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train_gru.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

train_cnn: train_cnn.c ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train_cnn.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

test: test.c ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. test.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

# predict.c reads a real WAV file through wav_reader.[ch]/audio_features.[ch]
# instead of data_prep.[ch]'s CSV rows, so it links those two instead.
predict: predict.c ../../audio_features.o ../../fft.o ../../wav_reader.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. predict.c ../../audio_features.o ../../fft.o ../../wav_reader.o ../../libnn.a $(LDFLAGS) -o $@

# Same two libraries as predict, for the same reason (real WAV files in,
# audio_features.[ch] doing the same feature extraction training will see).
prepare_data: prepare_data.c ../../audio_features.o ../../fft.o ../../wav_reader.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. prepare_data.c ../../audio_features.o ../../fft.o ../../wav_reader.o ../../libnn.a $(LDFLAGS) -o $@

# Downloads and extracts the real Google Speech Commands dataset (CC BY 4.0)
# -- ~2.4GB, one subdirectory per spoken word plus a "_background_noise_"
# directory of longer ambient recordings. `-z $@` makes curl a conditional
# fetch (skips re-downloading an already-current archive).
$(DATASET_ARCHIVE):
	curl -s -S -L -f $(DATASET_URL) -z $@ -o $@

$(DATASET_DIR)/$(WAKE_WORD): $(DATASET_ARCHIVE)
	mkdir -p $(DATASET_DIR)
	tar -xzf $(DATASET_ARCHIVE) -C $(DATASET_DIR)
	touch $@

# Unlike examples/character_recognition's samples.csv (MNIST's pixels,
# flattened and thresholded), flattening this dataset into feature rows is
# itself the thing this example exists to demonstrate -- see
# prepare_data.c's top comment. It writes all three splits itself, from the
# dataset's own speaker-separated validation/testing lists, instead of
# shuffling one samples.csv through ../../split.py.
$(CSV_OUTPUTS)&: prepare_data $(DATASET_DIR)/$(WAKE_WORD)
	./prepare_data $(DATASET_DIR) $(WAKE_WORD)

clean:
	$(RM) train_gru train_cnn test predict prepare_data model.* $(CSV_OUTPUTS)

distclean: clean
	$(RM) $(DATASET_ARCHIVE)
	$(RM) -r $(DATASET_DIR)
