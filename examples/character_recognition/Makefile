include ../../common.mk

CSV_OUTPUTS := test.csv train.csv validation.csv

.PHONY: all clean distclean

all: train_cnn train_fc test predict $(CSV_OUTPUTS)

# Recursing into the top-level Makefile (rather than duplicating nn.o's/
# data_prep.o's build rules here) keeps each compiled in exactly one place
# regardless of how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

../../data_prep.o:
	$(MAKE) -C ../.. data_prep.o

train_cnn: train_cnn.c ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train_cnn.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

train_fc: train_fc.c ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train_fc.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

test: test.c ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. test.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

# predict.c doesn't use data_prep.[ch] (it reads a single hardcoded sample,
# not train/test CSVs), so it links against ../../libnn.a alone.
predict: predict.c ../../libnn.a
	$(CC) $(CFLAGS) -I../.. predict.c ../../libnn.a $(LDFLAGS) -o $@

samples.csv:
	curl -s -S -L -f --compressed http://coledd.com/training_data/$@.gz -z $@ -o $@.gz && gunzip $@.gz

$(CSV_OUTPUTS)&: samples.csv ../../split.py
	python ../../split.py samples.csv --train 0.8 --validation 0.1

clean:
	$(RM) train_cnn train_fc test predict model.* $(CSV_OUTPUTS)

distclean: clean
	$(RM) samples.csv
