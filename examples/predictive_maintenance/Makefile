include ../../common.mk

.PHONY: all clean stm32

all: train test monitor

# Recursing into the top-level Makefile (rather than duplicating nn.o's/
# fft.o's build rules here) keeps each compiled in exactly one place
# regardless of how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

../../fft.o:
	$(MAKE) -C ../.. fft.o

# The on-device half: what ships to the STM32 alongside nn.c and fft.c.
DEVICE_OBJS := vibration_features.o health_monitor.o ../../fft.o
# The desktop-only stand-in for a real machine and its accelerometer.
SIM_OBJS := machine_sim.o

vibration_features.o: vibration_features.c vibration_features.h ../../fft.h
	$(CC) $(CFLAGS) -I../.. -c $<

health_monitor.o: health_monitor.c health_monitor.h vibration_features.h ../../nn.h
	$(CC) $(CFLAGS) -I../.. -c $<

machine_sim.o: machine_sim.c machine_sim.h health_monitor.h vibration_features.h
	$(CC) $(CFLAGS) -I../.. -c $<

train test monitor: %: %.c $(SIM_OBJS) $(DEVICE_OBJS) ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $< $(SIM_OBJS) $(DEVICE_OBJS) ../../libnn.a $(LDFLAGS) -o $@

# Cross-compiles the on-device half (plus nn.c and fft.c) for the STM32U585
# (Cortex-M33, single-precision FPU) with the GNU Arm Embedded toolchain
# and reports how much flash and RAM it takes -- a check that it really is
# firmware-ready, not a substitute for a board bring-up. Needs only the
# toolchain (STM32CubeIDE bundles one; point ARM_CC at it if it isn't on
# PATH), not the HAL. The model itself goes to flash separately; see
# README.md.
ARM_CC ?= arm-none-eabi-gcc
ARM_SIZE ?= $(patsubst %gcc,%size,$(ARM_CC))
ARM_CFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard -Os -ffunction-sections -fdata-sections -Wall -I../..
ARM_SOURCES := vibration_features.c health_monitor.c ../../fft.c ../../nn.c

stm32:
	mkdir -p stm32
	for f in $(ARM_SOURCES); do $(ARM_CC) $(ARM_CFLAGS) -c $$f -o stm32/$$(basename $${f%.c}).o || exit 1; done
	$(ARM_SIZE) -t stm32/*.o

clean:
	$(RM) $(SIM_OBJS) vibration_features.o health_monitor.o train test monitor model*.txt model*.bin model*.inplace
	$(RM) -r stm32
