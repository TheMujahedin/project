# ==============================================================================
# Makefile για το Σύστημα Γεωμετρικής Επαλήθευσης & Επανακατάταξης Εικόνων
# (SIFT + BoVW + Normalized DLT/SVD + RANSAC)
# ==============================================================================

# Ορισμός του C++ μεταγλωττιστή
CXX = g++

# Εντοπισμός include paths για Eigen3 και OpenCV 4
EIGEN_CFLAGS = $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)

# Σημαίες μεταγλώττισης (Compilation Flags):
# - -O3: Μέγιστη βελτιστοποίηση ταχύτητας για RANSAC και SIFT matching.
# - -Wall: Ενεργοποίηση προειδοποιήσεων του compiler.
# - -std=c++17: Πρότυπο C++17 (std::filesystem, structured bindings κ.λπ.).
CXXFLAGS = -O3 -Wall -std=c++17 $(EIGEN_CFLAGS) $(OPENCV_CFLAGS)

# Σημαίες σύνδεσης (Linker Flags)
LDFLAGS = $(shell pkg-config --libs opencv4)

# Το όνομα του παραγόμενου εκτελέσιμου αρχείου όπως ορίζεται στην εκφώνηση
TARGET = verify

# Λίστα όλων των πηγαίων αρχείων C++ (.cpp)
# Σημείωση: Τα ExactSearch.h και metrics.h είναι header-only
SRCS = main.cpp features.cpp geometry.cpp lsh.cpp hypercube.cpp ivfflat.cpp ivfpq.cpp

# Αυτόματη παραγωγή των αντικειμενικών αρχείων (.o)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean clean_all

# Προεπιλεγμένος στόχος: μεταγλώττιση και σύνδεση του εκτελέσιμου verify
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Κανόνας μεταγλώττισης κάθε .cpp σε αντίστοιχο .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Καθαρισμός των παραγόμενων αρχείων (.o, εκτελέσιμο verify, προεπιλεγμένη έξοδος)
clean:
	rm -f $(OBJS) $(TARGET) output.txt results.txt

# Πλήρης καθαρισμός (συμπεριλαμβανομένων προσωρινών αρχείων πειραμάτων & διαγραμμάτων)
clean_all: clean
	rm -f res_*.txt results_summary.csv *.png