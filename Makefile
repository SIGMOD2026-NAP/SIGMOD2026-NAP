SRCS=$(wildcard ./src/*.cpp)
OBJDIR=./build
RESDIR=./results
OBJS = $(patsubst ./src/%.cpp,$(OBJDIR)/%.o,$(SRCS))
TARGET = ht
CXX := g++
CXXFLAGS := -std=c++17 -mavx512f -Ofast -lrt -DNDEBUG  -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0


.PHONY:$(TARGET)

all: $(TARGET)

$(OBJDIR)/%.o:src/%.cpp
	@test -d $(OBJDIR) | mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET):$(OBJS)
	@test -d $(RESDIR) | mkdir -p $(RESDIR)
	@test -d ./indexes | mkdir -p ./indexes
	$(CXX) $(CXXFLAGS)  -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJDIR)
