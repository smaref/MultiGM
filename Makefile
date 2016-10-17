# Folders
BIN=bin
SRC=src
INC=include

# Compiler
NVCC = nvcc 
NVCCFLAGS = -O3 -std=c++11 -lineinfo
NVCCLIBS =  -lcudart -lcusparse -lcudart -lcublas -lcusparse

CXX = icpc
CXXFLAGS = -O3 -std=c++11
CXXLIBS = -mkl
all: distclean mkdir run_b run_bm run_s run_sm run_blocked_b run_blocked_bm run_blocked_s run_blocked_sm

build_all: distclean mkdir blas blas_m sparse sparse_m blas_blocked blas_m_blocked sparse_blocked sparse_m_blocked

mkd:
	mkdir -p $(BIN)

#cuSparse thrust, blocked affinity matrix, initially matched
sparse_thrust_m_blocked: $(SRC)/cuSparse_thrust_blocked_matched.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS)  

#cuBLAS, full affinity matrix
blas: $(SRC)/cuBLAS.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS) 

#cuBLAS, full affinity matrix, initially matched
blas_m: $(SRC)/cuBLAS_matched.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS) 

#cuBLAS, blocked affinity matrix
blas_blocked: $(SRC)/cuBLAS_blocked.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS) 

#cuBLAS, blocked affinity matrix, initially matched
blas_m_blocked: $(SRC)/cuBLAS_blocked_matched.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS) 


#cuSparse, full affinity matrix
sparse_m: $(SRC)/cuSparse_matched.cu mkd 
	$(NVCC) $(NVCCFLAGS)  $(NVCCLIBS) $< -I$(INC) -o $(BIN)/$@

#cuSparse, full affinity matrix, initially matched
#sparse_m: $(SRC)/cuSparse_matched.cu mkd 
#	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ 
#cuSparse, blocked affinity matrix
sparse_blocked: $(SRC)/cuSparse_blocked.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ 

#cuSparse, blocked affinity matrix, initially matched
sparse_m_blocked: $(SRC)/cuSparse_blocked_matched.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS) 

#mkl, blocked affinity matrix
mkl_blocked: $(SRC)/mkl_blocked.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#mkl, blocked affinity matrix, initially matched
mkl_m_blocked: $(SRC)/mkl_blocked_matched.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#mkl, full affinity matrix, initially matched
mkl_m: $(SRC)/mkl_matched.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)


run_blocked_stm: sparse_thrust_m_blocked
	$(BIN)/$< Input/img1 8 Input/img2 5 Input/matches.txt 10
	#$(BIN)/$< Input/building1_appr 683 Input/building2 1496 Input/building_matches 2000
run_b: blas
	$(BIN)/$< Input/ranch1_appr 62 Input/ranch2 66

run_bm: blas_m
	$(BIN)/$< Input/building1_appr 683 Input/building2 1496 Input/building_matches 2000
#	$(BIN)/$< Input/img1 8 Input/img2 5 Input/matches.txt 10

run_s: sparse
	$(BIN)/$< Input/img1 8 Input/img2 5 

run_sm: sparse_m
	$(BIN)/$< Input/img1 8 Input/img2 5 Input/matches.txt 10

run_blocked_b: blas_blocked
	time $(BIN)/$< Input/bike1_appr 292 Input/bike2 703
	#$(BIN)/$< Input/building1_appr 683 Input/building2 1496

run_blocked_bm: blas_m_blocked
	$(BIN)/$< Input/img1 8 Input/img2 5 Input/matches.txt 10
#	$(BIN)/$< Input/building1_appr 683 Input/building2 1496 Input/building_matches 2000

run_blocked_s: sparse_blocked
	$(BIN)/$< Input/img1 8 Input/img2 5

run_blocked_sm: sparse_m_blocked
	$(BIN)/$< Input/img1 8 Input/img2 5 Input/matches.txt 10

run_blocked_mm: mkl_m_blocked
#	$(BIN)/$< Input/img1 8 Input/img2 5 Input/matches.txt 10

run_mm: mkl_m
	$(BIN)/$< Input/building1_appr 683 Input/building2 1496 Input/building_matches 2000
#	$(BIN)/$< Input/img1 8 Input/img2 5 Input/matches.txt 10

run_blocked_m: mkl_blocked
	time $(BIN)/$< Input/bike1_appr 292 Input/bike2 703
#   $(BIN)/$< Input/building1_appr 683 Input/building2 1496
#	$(BIN)/$< Input/building1_appr 683 Input/building2 1496 Input/building_matches 2000
### with initial matching
#$(BIN)/$< Input/ranch1_appr 62 Input/ranch2 66 Input/ranch_matches 621
#$(BIN)/$< Input/bike1_appr 292 Input/bike2 703 Input/bike_matches 2000  
#$(BIN)/$< Input/building1_appr 683 Input/building2 1496 Input/building_matches 2000

### without initial matching
#$(BIN)/$< Input/ranch1_appr 62 Input/ranch2 66
#$(BIN)/$< Input/bike1_appr 292 Input/bike2 703
#$(BIN)/$< Input/building1_appr 683 Input/building2 1496

clean:
	rm -if $(BIN)/*

distclean:
	rm -irf $(BIN)
