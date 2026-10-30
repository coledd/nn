include ../../common.mk

CSV_OUTPUTS := test.csv train.csv validation.csv
DATASET_DIR := dataset
# MNIST (LeCun, Cortes & Burges; CC BY-SA 3.0): the four original IDX
# files, ~11MB compressed. yann.lecun.com no longer serves them, so they
# come from Google's CVDF mirror (the one TensorFlow's MNIST loaders use),
# whose files are byte-identical to the originals -- the same MD5s
# torchvision checks against.
MNIST_URL := https://storage.googleapis.com/cvdf-datasets/mnist
MNIST_FILES := $(addprefix $(DATASET_DIR)/,train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte)

.PHONY: all clean distclean

all: train_cnn train_fc test predict $(CSV_OUTPUTS)

# Recursing into the top-level Makefile (rather than duplicating nn.o's/
# data_prep.o's build rules here) keeps each compiled in exactly one place
# regardless of how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

../../data_prep.o:
	$(MAKE) -C ../.. data_prep.o

train_cnn: train_cnn.c augment.h ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train_cnn.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

train_fc: train_fc.c augment.h ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train_fc.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

test: test.c ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. test.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

# prepare_data.c reads MNIST's own file format directly, so it needs
# neither the library nor data_prep.[ch].
prepare_data: prepare_data.c
	$(CC) $(CFLAGS) prepare_data.c $(LDFLAGS) -o $@

# predict.c doesn't use data_prep.[ch] (it reads a single hardcoded sample,
# not train/test CSVs), so it links against ../../libnn.a alone.
predict: predict.c ../../libnn.a
	$(CC) $(CFLAGS) -I../.. predict.c ../../libnn.a $(LDFLAGS) -o $@

$(MNIST_FILES): $(DATASET_DIR)/%:
	mkdir -p $(DATASET_DIR)
	curl -s -S -L -f $(MNIST_URL)/$*.gz -o $@.gz
	gunzip -f $@.gz

# prepare_data writes all three CSVs, keeping MNIST's own train/test split
# (see its top comment).
$(CSV_OUTPUTS)&: prepare_data $(MNIST_FILES)
	./prepare_data $(DATASET_DIR)

clean:
	$(RM) train_cnn train_fc test predict prepare_data model.* $(CSV_OUTPUTS)

distclean: clean
	$(RM) -r $(DATASET_DIR)
