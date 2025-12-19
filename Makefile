# ------------------------- Compiler Settings ------------------------- #
CXX := g++

SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build

TARGET_NAME := cpp_eigen_opencv

EIGEN_CFLAGS	:= $(shell pkg-config --cflags eigen3)
EIGEN_ISYSTEM	:= $(patsubst -I%,-isystem %,$(EIGEN_CFLAGS))
EIGEN_LIBS		:= $(shell pkg-config --libs eigen3)

OPENCV_CFLAGS   := $(shell pkg-config --cflags opencv4)
OPENCV_ISYSTEM	:= $(patsubst -I%,-isystem %,$(OPENCV_CFLAGS))
OPENCV_LIBS     := $(shell pkg-config --libs opencv4)


# -Weffc++ -Wconversion -Wsign-conversion -pedantic-errors
# -Werror
COMMON_CXXFLAGS := -std=c++23 -fdiagnostics-color=always -pedantic-errors \
				   -Wall -Wextra -Werror \
				   -Weffc++ -Wconversion -Wsign-conversion \
				   -I$(INC_DIR) $(OPENCV_ISYSTEM) $(EIGEN_ISYSTEM)

# -DEIGEN_NO_DEBUG (add only if debug performance is too bad)
DEBUG_CXXFLAGS      := -O0 -g -ggdb -DDEBUG -fno-omit-frame-pointer
ASAN_CXXFLAGS		:= $(DEBUG_CXXFLAGS) -fsanitize=address,undefined
RELEASE_CXXFLAGS    := -O3 -DNDEBUG -march=native


# -fsanitize=address,undefined
DEBUG_LDFLAGS   :=
ASAN_LDFLAGS	:= $(DEBUG_LDFLAGS) -fsanitize=address,undefined
RELEASE_LDFLAGS :=

# ------------------------- Source Discovery ------------------------- #

SRCS    := $(shell find $(SRC_DIR) -name '*.cpp')

# ------------------------- DEBUG ------------------------- #

DEBUG_BUILD_DIR := $(BUILD_DIR)/debug
DEBUG_OBJS      := $(patsubst $(SRC_DIR)/%.cpp,$(DEBUG_BUILD_DIR)/%.o,$(SRCS))
DEBUG_TARGET    := $(DEBUG_BUILD_DIR)/$(TARGET_NAME)

# ------------------------- DEBUG ------------------------- #

ASAN_BUILD_DIR := $(BUILD_DIR)/asan
ASAN_OBJS      := $(patsubst $(SRC_DIR)/%.cpp,$(ASAN_BUILD_DIR)/%.o,$(SRCS))
ASAN_TARGET    := $(ASAN_BUILD_DIR)/$(TARGET_NAME)

# ------------------------- RELEASE ------------------------- #

RELEASE_BUILD_DIR   := $(BUILD_DIR)/release
RELEASE_OBJS        := $(patsubst $(SRC_DIR)/%.cpp,$(RELEASE_BUILD_DIR)/%.o,$(SRCS))
RELEASE_TARGET      := $(RELEASE_BUILD_DIR)/$(TARGET_NAME)

# ------------------------- Default ------------------------- #

all: rel

# ------------------------- Targets ------------------------- #

dbg:	$(DEBUG_TARGET)
asan:	$(ASAN_TARGET)
rel:	$(RELEASE_TARGET)

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CXX) $^ -o $@ $(OPENCV_LIBS) $(EIGEN_LIBS) $(DEBUG_LDFLAGS)

$(ASAN_TARGET): $(ASAN_OBJS)
	$(CXX) $^ -o $@ $(OPENCV_LIBS) $(EIGEN_LIBS) $(ASAN_LDFLAGS)

$(RELEASE_TARGET): $(RELEASE_OBJS)
	$(CXX) $^ -o $@ $(OPENCV_LIBS) $(EIGEN_LIBS) $(RELEASE_LDFLAGS)

# ------------------------- Compilation Rules ------------------------- #

$(DEBUG_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(COMMON_CXXFLAGS) $(DEBUG_CXXFLAGS) -c $< -o $@

$(ASAN_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(COMMON_CXXFLAGS) $(ASAN_CXXFLAGS) -c $< -o $@

$(RELEASE_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(COMMON_CXXFLAGS) $(RELEASE_CXXFLAGS) -c $< -o $@

# ------------------------- Cleanup ------------------------- #

clean:
	rm -rf $(BUILD_DIR)

# ------------------------- PHONY ------------------------- #

.PHONY: all dbg asan rel clean
