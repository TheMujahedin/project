# ==============================================================================
# Makefile για το Σύστημα Ανάκτησης Εικόνων (Image Retrieval System)
# 
# Υποστηριζόμενοι αλγόριθμοι:
# - Exact Nearest Neighbor Search (Ground Truth)
# - Locality Sensitive Hashing (LSH)
# - Random Hypercube Projection
# - Inverted File with Flat Vectors (IVFFlat)
# - Inverted File with Product Quantization (IVFPQ)
# ==============================================================================

# Ορισμός του C++ μεταγλωττιστή
CXX = g++

# Σημαίες μεταγλώττισης (Compilation Flags):
# - -O3: Μέγιστη επιθετική βελτιστοποίηση κώδικα για υψηλή ταχύτητα εκτέλεσης.
# - -Wall: Ενεργοποίηση όλων των προειδοποιήσεων (warnings) του compiler.
# - -std=c++17: Χρήση του προτύπου C++17 (απαιτείται για σύγχρονες δομές και lambdas).
# - pkg-config --cflags: Αυτόματη ανάκτηση των include directories της OpenCV 4.
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)

# Σημαίες σύνδεσης (Linker Flags):
# - pkg-config --libs: Σύνδεση με τις απαραίτητες βιβλιοθήκες της OpenCV (core, imgcodecs, features2d).
LDFLAGS = $(shell pkg-config --libs opencv4)

# Το όνομα του παραγόμενου εκτελέσιμου αρχείου
TARGET = search

# Λίστα όλων των πηγαίων αρχείων C++ (.cpp)
# Σημείωση: Τα ExactSearch.h και metrics.h είναι header-only και δεν χρειάζονται ξεχωριστό .cpp
SRCS = main.cpp features.cpp lsh.cpp hypercube.cpp ivfflat.cpp ivfpq.cpp

# Αυτόματη παραγωγή των ονομάτων των αντικειμενικών αρχείων (.o) από τα αντίστοιχα .cpp
OBJS = $(SRCS:.cpp=.o)

# Δήλωση ψευδο-στόχων (Phony Targets) για αποφυγή συγκρούσεων με τυχόν ομώνυμα αρχεία στο δίσκο
.PHONY: all clean clean_all

# Προεπιλεγμένος στόχος: μεταγλώττιση και σύνδεση του εκτελέσιμου
all: $(TARGET)

# Κανόνας σύνδεσης (Linking) των αντικειμενικών αρχείων (.o) για τη δημιουργία του εκτελέσιμου
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Κανόνας μοτίβου (Pattern Rule) για τη μεταγλώττιση κάθε .cpp σε αντίστοιχο .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Κανόνας καθαρισμού των παραγόμενων αρχείων (.o, εκτελέσιμο, output.txt)
clean:
	rm -f $(OBJS) $(TARGET) output.txt

# Κανόνας πλήρους καθαρισμού (περιλαμβάνει και τα προσωρινά αρχεία αποτελεσμάτων πειραμάτων)
clean_all: clean
	rm -f res_*.txt results_summary.csv *.png