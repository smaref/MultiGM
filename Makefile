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


#matrix vector multipllication
original: $(SRC)/original.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ 

run_orig: original
	$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > outorig.text



#/*********************************** MKL ************************************/
#mkl, full affinity matrix
mkl: $(SRC)/mkl_orig.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#mkl, full affinity matrix, initially matched
mkl_m: $(SRC)/mkl_matched.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#mkl, blocked affinity matrix
mkl_blocked: $(SRC)/mkl_blocked.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#mkl, blocked affinity matrix, initially matched
mkl_m_blocked: $(SRC)/mkl_blocked_match.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#.............................................................................#
run_m: mkl
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/mkl1_xs.text
	$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/mkl1_s.text
#	$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/mkl1_m.text

	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_mkl1.text
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_mkl1.text

run_mm: mkl_m
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 Inp/xsmall/matches.text 87 > output/mkl2_xs.text
	#$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 Inp/small/matches.text 227 > output/mkl2_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 Inp/medium/matches.text 270 > output/mkl2_m.text
	$(BIN)/$< Inp/xlarge/dist1.text 299 Inp/xlarge/dist2.text 196 Inp/xlarge/dist3.text 162 Inp/xlarge/matches.text 7704 > output/mkl2_xl.text
	
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 Input/mat.text 6 > out_mkl2.text
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 rioData/matches.text 1999 > outRio_mkl2.text
	#$(BIN)/$< Input/dist1.text 9 Input/dist2.text 5 Input/dist3.text 4 Input/matches.text 10 > out_mkl2.text

run_blocked_m: mkl_blocked
	#$(BIN)/$< Inp/xsmall/dist1_bl.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/mkl3_xs.text
	$(BIN)/$< Inp/small/dist1_bl.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/mkl3_s.text
	#$(BIN)/$< Inp/medium/dist1_bl.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/mkl3_m.text
	
	#time $(BIN)/$< Input/bike1_appr 292 Input/bike2 703
	#$(BIN)/$< rioData/dist1_bl.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_mkl3.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_mkl3.text 

run_blocked_mm: mkl_m_blocked
	#$(BIN)/$< Inp/xsmall/dist1_blm.text 19 Inp/xsmall/dist2_blm.text 19 Inp/xsmall/dist3_blm.text 23 > output/mkl4_xs.text
	$(BIN)/$< Inp/small/dist1_blm.text 52 Inp/small/dist2_blm.text 46 Inp/small/dist3_blm.text 46 > output/mkl4_s.text
	#$(BIN)/$< Inp/medium/dist1_blm.text 76 Inp/medium/dist2_blm.text 62 Inp/medium/dist3_blm.text 57 > output/mkl4_m.text
	
	#$(BIN)/$< rioData/dist1_blm.text 983 rioData/dist2_blm.text 491 rioData/dist3_blm.text 437 > outRio/out_mkl4.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_mkl4.text 


#/*********************************** MKL-COO ************************************/
#mkl_coo, sparse affinity matrix
mkl_coo: $(SRC)/mkl_coo_orig.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#mkl, full affinity matrix, initially matched
mkl_m_coo: $(SRC)/mkl_coo_match.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

mkl_bl_coo: $(SRC)/mkl_coo_blocked.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

mkl_blm_coo: $(SRC)/mkl_coo_blocked_match.cpp mkd 
	$(CXX) $(CXXFLAGS) $< -I$(INC) -I/usr/include -o $(BIN)/$@ $(CXXLIBS)

#.............................................................................#
run_mc: mkl_coo
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/mklCoo1_xs.text
	$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/mklCoo1_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/mklCoo1_m.text
	
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_mklCoo1.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_mkl_coo1.text

run_mmc: mkl_m_coo
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 Inp/xsmall/matches.text 87 > output/mklCoo2_xs.text
	#$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 Inp/small/matches.text 227 > output/mklCoo2_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 Inp/medium/matches.text 270 > output/mklCoo2_m.text
	$(BIN)/$< Inp/xlarge/dist1.text 299 Inp/xlarge/dist2.text 196 Inp/xlarge/dist3.text 162 Inp/xlarge/matches.text 7704 > output/mklCoo2_xl.text
	
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 rioData/matches.text 1999> outRio/out_mklCoo2.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 Input/mat.text 6 
	#$(BIN)/$< Input/dist1.text 9 Input/dist2.text 5 Input/dist3.text 4 Input/matches.text 10 > out_mkl_coo2.text

run_bl_mc: mkl_bl_coo
	#$(BIN)/$< Inp/xsmall/dist1_bl.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/mklCoo3_xs.text
	$(BIN)/$< Inp/small/dist1_bl.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/mklCoo3_s.text
	#$(BIN)/$< Inp/medium/dist1_bl.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/mklCoo3_m.text
	
	#$(BIN)/$< rioData/dist1_bl.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_mklCoo3.text
	#$(BIN)/$< Input/dist1.text 9 Input/dist2.text 5 Input/dist3.text 4 > out_mkl_coo3.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_mkl_coo3.text

run_bl_mmc: mkl_blm_coo
	#$(BIN)/$< Inp/xsmall/dist1_blm.text 19 Inp/xsmall/dist2_blm.text 19 Inp/xsmall/dist3_blm.text 23 > output/mklCoo4_xs.text
	$(BIN)/$< Inp/small/dist1_blm.text 52 Inp/small/dist2_blm.text 46 Inp/small/dist3_blm.text 46 > output/mklCoo4_s.text
	#$(BIN)/$< Inp/medium/dist1_blm.text 76 Inp/medium/dist2_blm.text 62 Inp/medium/dist3_blm.text 57 > output/mklCoo4_m.text
	
	#$(BIN)/$< rioData/dist1_blm.text 983 rioData/dist2_blm.text 491 rioData/dist3_blm.text 437 > outRio/out_mklCoo4.text
	#$(BIN)/$< Input/dist1.text 9 Input/dist2.text 5 Input/dist3.text 4 > out_mkl_coo4.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_mkl_coo4.text


#/****************************** cuBlas **************************************/
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

#.............................................................................#
run_b: blas
	$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/blas1_xs.text
	#$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/blas1_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/blas1_m.text
#	
#	$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_blas1.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_blas1.text

run_bm: blas_m
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 Inp/xsmall/matches.text 87 > output/blas2_xs.text
	#$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 Inp/small/matches.text 227 > output/blas2_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 Inp/medium/matches.text 270 > output/blas2_m.text
	$(BIN)/$< Inp/xlarge/dist1.text 299 Inp/xlarge/dist2.text 196 Inp/xlarge/dist3.text 162 Inp/xlarge/matches.text 7704 > output/blas2_xl.text
	
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 rioData/matches.text 1999> outRio/out_blas2.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 Input/mat.text 6 > out_blas2.text

run_blocked_b: blas_blocked
	$(BIN)/$< Inp/xsmall/dist1_bl.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/blas3_xs.text
	#$(BIN)/$< Inp/small/dist1_bl.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/blas3_s.text
	#$(BIN)/$< Inp/medium/dist1_bl.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/blas3_m.text
	
	#$(BIN)/$< rioData/dist1_bl.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_bls3.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_blas3.text

run_blocked_bm: blas_m_blocked
	#$(BIN)/$< Inp/xsmall/dist1_blm.text 19 Inp/xsmall/dist2_blm.text 19 Inp/xsmall/dist3_blm.text 23 > output/blas4_xs.text
	$(BIN)/$< Inp/small/dist1_blm.text 52 Inp/small/dist2_blm.text 46 Inp/small/dist3_blm.text 46 > output/blas4_s.text
	#$(BIN)/$< Inp/medium/dist1_blm.text 76 Inp/medium/dist2_blm.text 62 Inp/medium/dist3_blm.text 57 > output/blas4_m.text
	
	#$(BIN)/$< rioData/dist1_blm.text 983 rioData/dist2_blm.text 491 rioData/dist3_blm.text 437 > outRio/out_bls4.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_blas4.text



#/********************** cuSparse COO affinity matrix ************************/
#cuSparse COO, original affinity matrix
sparse_coo: $(SRC)/cuSparse_coo_orig.cu mkd 
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBS) $< -I$(INC) -o $(BIN)/$@

#cuSparse COO, initially matches affinity matrix
sparse_coo_m: $(SRC)/cuSparse_coo_match.cu mkd 
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBS) $< -I$(INC) -o $(BIN)/$@

#cuSparse COO, blocked affinity matrix
sparse_coo_blocked: $(SRC)/cuSparse_coo_blocked.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS)  

#cuSparse COO, blocked affinity matrix, initially matched
sparse_coo_m_blocked: $(SRC)/cuSparse_coo_blocked_match.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS)  

#cuSparse COO, blocked affinity matrix, initially matched
sparse_thrust_coo_m_blocked: $(SRC)/cuSparse_thrust_coo_block_match.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS)  

#.............................................................................#
run_cs: sparse_coo
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/sCoo1_xs.text
	$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/sCoo1_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/sCoo1_m.text
	
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_scoo1.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_coo1.text

run_csm: sparse_coo_m
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 Inp/xsmall/matches.text 87 > output/sCoo2_xs.text
	#$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 Inp/small/matches.text 227 > output/sCoo2_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 Inp/medium/matches.text 270 > output/sCoo2_m.text
	$(BIN)/$< Inp/xlarge/dist1.text 299 Inp/xlarge/dist2.text 196 Inp/xlarge/dist3.text 162 Inp/xlarge/matches.text 7704 > output/sCoo2_xl.text
	
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 rioData/matches.text 1999> outRio/out_scoo2.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 Input/mat.text 6 > out_coo2.text

run_blocked_cs: sparse_coo_blocked
	#$(BIN)/$< Inp/xsmall/dist1_bl.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/sCoo3_xs.text
	$(BIN)/$< Inp/small/dist1_bl.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/sCoo3_s.text
	#$(BIN)/$< Inp/medium/dist1_bl.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/sCoo3_m.text
	
	#$(BIN)/$< rioData/dist1_bl.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_sCoo3.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_coo3.text

run_blocked_csm: sparse_coo_m_blocked
	#$(BIN)/$< Inp/xsmall/dist1_blm.text 19 Inp/xsmall/dist2_blm.text 19 Inp/xsmall/dist3_blm.text 23 > output/sCoo4_xs.text
	$(BIN)/$< Inp/small/dist1_blm.text 52 Inp/small/dist2_blm.text 46 Inp/small/dist3_blm.text 46 > output/sCoo4_s.text
	#$(BIN)/$< Inp/medium/dist1_blm.text 76 Inp/medium/dist2_blm.text 62 Inp/medium/dist3_blm.text 57 > output/sCoo4_m.text
	
	#$(BIN)/$< rioData/dist1_blm.text 983 rioData/dist2_blm.text 491 rioData/dist3_blm.text 437 > outRio/out_sCoo4.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_coo4.text



#/******************** cuSparse CSR affinity matrix  *******************/
#cuSparse CSR, initially matches affinity matrix
sparse_m: $(SRC)/cuSparse_matched.cu mkd 
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBS) $< -I$(INC) -o $(BIN)/$@

#cuSparse CSR, blocked affinity matrix
sparse_blocked: $(SRC)/cuSparse_blocked.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS)  

#cuSparse CSR, blocked affinity matrix, initially matched
sparse_m_blocked: $(SRC)/cuSparse_blocked_matched.cu mkd 
	$(NVCC) $(NVCCFLAGS) $< -I$(INC) -o $(BIN)/$@ $(NVCCLIBS) 

#.............................................................................#
run_sm: sparse_m
	#$(BIN)/$< Inp/xsmall/dist1.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 Inp/xsmall/matches.text 87 > output/sparse2_xs.text
	#$(BIN)/$< Inp/small/dist1.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 Inp/small/matches.text 227 > output/sparse2_s.text
	#$(BIN)/$< Inp/medium/dist1.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 Inp/medium/matches.text 270 > output/sparse2_m.text
	$(BIN)/$< Inp/xlarge/dist1.text 299 Inp/xlarge/dist2.text 196 Inp/xlarge/dist3.text 162 Inp/xlarge/matches.text 7704 > output/sparse2_xl.text
	
	#$(BIN)/$< rioData/dist1.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 rioData/matches.text 1999 > outRio/out_sparse2.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 Input/mat.text 6 > out_sparse2.text

run_blocked_s: sparse_blocked
	#$(BIN)/$< Inp/xsmall/dist1_bl.text 20 Inp/xsmall/dist2.text 27 Inp/xsmall/dist3.text 32 > output/sparse3_xs.text
	$(BIN)/$< Inp/small/dist1_bl.text 58 Inp/small/dist2.text 62 Inp/small/dist3.text 67 > output/sparse3_s.text
	#$(BIN)/$< Inp/medium/dist1_bl.text 89 Inp/medium/dist2.text 78 Inp/medium/dist3.text 79 > output/sparse3_m.text
	
	#$(BIN)/$< rioData/dist1_bl.text 983 rioData/dist2.text 984 rioData/dist3.text 1178 > outRio/out_sparse3.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_sparse3.text

run_blocked_sm: sparse_m_blocked
	#$(BIN)/$< Inp/xsmall/dist1_blm.text 19 Inp/xsmall/dist2_blm.text 19 Inp/xsmall/dist3_blm.text 23 > output/sparse4_xs.text
	$(BIN)/$< Inp/small/dist1_blm.text 52 Inp/small/dist2_blm.text 46 Inp/small/dist3_blm.text 46 > output/sparse4_s.text
	#$(BIN)/$< Inp/medium/dist1_blm.text 76 Inp/medium/dist2_blm.text 62 Inp/medium/dist3_blm.text 57 > output/sparse4_m.text
	
	#$(BIN)/$< rioData/dist1_blm.text 983 rioData/dist2_blm.text 491 rioData/dist3_blm.text 437 > outRio/out_sparse4.text
	#$(BIN)/$< Input/inp1.text 6 Input/inp2.text 4 Input/inp3.text 5 > out_sparse4.text






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
