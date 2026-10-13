include common.mk

.PHONY: all clean check tags

# The embeddable library (nn.[ch]) and the general-purpose model tools that
# operate on any saved model regardless of which example produced it.
# Each examples/*/ directory has its own, separate Makefile (see
# examples/*/Makefile) -- run `make` there to build that example; it isn't
# built by this one.
all: prune quantize dequantize summary export import libnn.a libnn.so

libnn.a: nn.o
	$(RM) $@
	$(AR) rv $@ $^

libnn.so: nn.o
	$(RM) $@
	$(AR) rcs $@ $^
	$(CC) -shared -Wl,-soname,libnn.so -o $@ $^

nn.o: nn.c nn.h
	$(CC) $(CFLAGS) -c $<

# General-purpose CSV data loading (read/parse/shuffle flat input/target
# rows), reusable by any example that wants CSV-backed training data --
# unlike gesture_data.o/fall_data.o (each specific to one example's own
# synthetic generator), this has no domain-specific logic of its own. Not
# part of libnn.a for the same reason as those two: it's a desktop-side
# training-data utility, not something an embedded target would want built
# into the library. No root tool uses it directly, so it isn't part of
# `all` -- examples/*/Makefile recurses into `$(MAKE) -C ../.. data_prep.o`
# for whichever example needs it (currently just character_recognition).
data_prep.o: data_prep.c data_prep.h
	$(CC) $(CFLAGS) -c $<

# On-device-reusable audio feature extraction (framing/FFT/mel
# filterbank/log-energy). Unlike data_prep.o above, this DOES need to ship
# into firmware alongside nn.o: a keyword-spotting target has to run the
# exact same feature extraction against its live mic buffer that prepared
# its training data, or the model sees different features at inference than
# it was trained on. Still not part of libnn.a (it isn't neural-net code)
# or of `all` (no root tool uses it directly) -- examples/*/Makefile
# recurses into `$(MAKE) -C ../.. audio_features.o` for whichever example
# needs it (currently just wake_word_detection).
audio_features.o: audio_features.c audio_features.h fft.h
	$(CC) $(CFLAGS) -c $<

# The radix-2 FFT behind both on-device feature extractors (audio_features.o
# above and examples/predictive_maintenance's vibration_features.o), kept
# here so neither carries its own copy. Ships into firmware with whichever
# of them a target uses; not part of libnn.a or `all`, same reasoning as
# audio_features.o -- examples recurse into `$(MAKE) -C ../.. fft.o`.
fft.o: fft.c fft.h
	$(CC) $(CFLAGS) -c $<

# Desktop-only 16-bit PCM mono WAV decoder -- same category as data_prep.o
# above (format-agnostic, no dataset-specific logic), but, like data_prep.o,
# not something an embedded target wants: firmware gets PCM straight from
# its own mic/ADC driver, never a .wav file. Not part of libnn.a or `all`,
# same reasoning as data_prep.o.
wav_reader.o: wav_reader.c wav_reader.h
	$(CC) $(CFLAGS) -c $<

prune: prune.c libnn.a
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

summary: summary.c libnn.a
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

quantize: quantize.c libnn.a
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

dequantize: dequantize.c libnn.a
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

export: export.c libnn.a
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

import: import.c libnn.a
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) nn.o data_prep.o audio_features.o fft.o wav_reader.o libnn.a libnn.so prune quantize dequantize summary export import tags
	for d in examples/*/; do $(MAKE) -C "$$d" clean; done
