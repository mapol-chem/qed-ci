#include <stdlib.h>
#include <stdio.h>
#include "orbital.h"
#include <string.h>
#include <math.h>
#include <mkl.h>
#include<omp.h>
#include<time.h>
#include <unistd.h>


void full_transformation_macroiteration(double* U, double* h2e, double* J, double *K, int* index_map_pq, int* index_map_kl, int nmo, int n_occupied) {
    size_t nmo_t = (size_t) nmo;
    size_t n_occupied_t = (size_t) n_occupied;
    double* h2e_half = (double*) malloc((size_t) nmo_t *(nmo_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    memset(h2e_half, 0, nmo_t *(nmo_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    double* temp1 = (double*) malloc((size_t) nmo_t *(nmo_t+1)/2 * nmo_t * n_occupied_t * sizeof(double));
    memset(temp1, 0, nmo_t *(nmo_t+1)/2 * nmo_t * n_occupied_t * sizeof(double));
    double* temp2 = (double*) malloc((size_t) nmo_t *(nmo_t+1)/2 * nmo_t * n_occupied_t * sizeof(double));
    memset(temp2, 0, nmo_t *(nmo_t+1)/2 * nmo_t * n_occupied_t * sizeof(double));
    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    print("%4d%4d%4d\n", pq_up, p , q);
    //}


    #pragma omp parallel for num_threads(16)
    for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
	size_t stride = pq_up * 2;
        size_t p = index_map_pq[stride];
        size_t q = index_map_pq[stride+1];
	//size_t pq = p * nmo_t + q;
	//print("%4d%4d%4d\n", pq_up, p , q);
        for (size_t r = 0; r < nmo_t; r++) {
            for (size_t s = 0; s < nmo_t; s++) {
		size_t rs = r * nmo_t + s;
		size_t pr = p * nmo_t + r;
		size_t qs = q * nmo_t + s;
	        h2e_half[pq_up * nmo_t * nmo_t + rs] = h2e[pr * nmo_t * nmo_t + qs];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
        size_t stride = pq_up * 2;
        size_t p = index_map_pq[stride];
        size_t q = index_map_pq[stride+1];
	size_t pq = p * nmo_t + q;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, 1.0, h2e+pq*nmo_t*nmo_t,
                  nmo_t, U, nmo_t, 0.0,
                  temp1+pq_up*nmo_t*n_occupied_t, n_occupied_t);
    }

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //    	    a += h2e[pq * nmo_t * nmo_t + rs] * U[s * nmo_t + l];
    //            }
    //    	temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] = a;
    //        }
    //    }
    //}
    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l], temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //            print("%20.12lf \n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] - temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //        }
    //    }
    //}


    double* temp3 = (double*) malloc((size_t) nmo_t *(nmo_t+1)/2 * n_occupied_t * n_occupied_t * sizeof(double));
    memset(temp3, 0, nmo_t *(nmo_t+1)/2 * n_occupied_t * n_occupied_t * sizeof(double));
    //double* temp4 = (double*) malloc(nmo_t *(nmo_t+1)/2 * n_occupied_t * n_occupied_t * sizeof(double));
    //memset(temp4, 0, nmo_t *(nmo_t+1)/2 * n_occupied_t * n_occupied_t * sizeof(double));
    // when matrix is transposed, in cblas_dgemm, m,n,k are the physical size of transposed matrix
    // but lda,ldb are leading dimensions (strides to next row) of original matrices (number of columns)
    #pragma omp parallel for num_threads(16)
    for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
        //size_t stride = pq_up * 2;
        //size_t p = index_map_pq[stride];
        //size_t q = index_map_pq[stride+1];
	//size_t pq = p * nmo_t + q;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, nmo_t, 1.0, U,
                  nmo_t, temp1+pq_up*nmo_t*n_occupied_t, n_occupied_t, 0.0,
                  temp3+pq_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t r = 0; r < nmo_t; r++) {
    //                size_t rl = r * n_occupied_t + l;
    //    	    a += temp1[pq_up * nmo_t * n_occupied_t + rl] * U[r * nmo_t + k];
    //            }
    //    	temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //        }
    //    }
    //}

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l], temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //            print("%20.12lf \n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]- temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //        }	
    //    }
    //}
    double* temp5 = (double*) malloc((size_t)n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    memset(temp5, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    double* temp6 = (double*) malloc((size_t)n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    memset(temp6, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    temp5[kl_up * nmo_t * nmo_t + pq] = temp3[pq_up * n_occupied_t * n_occupied_t + kl];
	    temp5[kl_up * nmo_t * nmo_t + qp] = temp5[kl_up * nmo_t * nmo_t + pq];
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, temp5+kl_up*nmo_t*nmo_t,
                  nmo_t, U, nmo_t, 0.0,
                  temp6+kl_up*nmo_t*nmo_t, nmo_t);
    }
    //double* temp7 = (double*) malloc(n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    //memset(temp7, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t q = 0; q < nmo_t; q++) {
    //                size_t pq = p * nmo_t + q;
    //    	    a += temp5[kl_up * nmo_t * nmo_t + pq] * U[q * nmo_t + s];
    //            }
    //    	temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] = a;
    //        }
    //    }
    //}
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s], temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //            print("%20.12lf \n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] - temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //        }
    //    }
    //}
    double* temp8 = (double*) malloc((size_t)n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    memset(temp8, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    //double* temp9 = (double*) malloc(n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    //memset(temp9, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp6+kl_up*nmo_t*nmo_t, nmo_t, 0.0,
                  temp8+kl_up*nmo_t*nmo_t, nmo_t);
    }
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t p = 0; p < nmo_t; p++) {
    //                size_t ps = p * nmo_t + s;
    //    	    a += temp6[kl_up * nmo_t * nmo_t + ps] * U[p * nmo_t + r];
    //            }
    //    	temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] = a;
    //        }
    //    }
    //}
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s], temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //            print("%20.12lf \n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] - temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //        }
    //    }
    //}
    //double* temp10 = (double*) malloc(nmo_t * nmo_t * nmo_t * n_occupied_t* sizeof(double));
    //memset(temp10, 0,  nmo_t * nmo_t * nmo_t * n_occupied_t * sizeof(double));
    //double* temp11 = (double*) malloc( nmo_t * nmo_t * n_occupied_t * n_occupied_t* sizeof(double));
    //memset(temp11, 0, nmo_t * nmo_t * n_occupied_t * n_occupied_t * sizeof(double));
    //double* temp12 = (double*) malloc( nmo_t * nmo_t * n_occupied_t * n_occupied_t* sizeof(double));
    //memset(temp12, 0, nmo_t * nmo_t * n_occupied_t * n_occupied_t * sizeof(double));
    //double* temp13 = (double*) malloc( nmo_t * nmo_t * n_occupied_t * n_occupied_t* sizeof(double));
    //memset(temp13, 0, nmo_t * nmo_t * n_occupied_t * n_occupied_t * sizeof(double));
    //fflush(stdout);
 

    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    J[kl * nmo_t * nmo_t + pq] = temp8[kl_up * nmo_t * nmo_t + pq];
	    J[kl * nmo_t * nmo_t + qp] = J[kl * nmo_t * nmo_t + pq];
	    J[lk * nmo_t * nmo_t + pq] = J[kl * nmo_t * nmo_t + pq];
	    J[lk * nmo_t * nmo_t + qp] = J[kl * nmo_t * nmo_t + pq];
	}
    }
    //print("test full transformation\n");
    //for (size_t p = 0; p < nmo_t; p++) {
    //    for (size_t q = 0; q < nmo_t; q++) {
    //        size_t pq = p * nmo_t + q;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    double a = 0.0;
    //                for (size_t s = 0; s < nmo_t; s++) {
    //    	        size_t rs = r * nmo_t + s;
    //                    a += h2e[pq * nmo_t * nmo_t + rs] * U[s * nmo_t + l];
    //    	    }	
    //                temp10[pq * nmo_t * n_occupied_t + r * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}
    //for (size_t p = 0; p < nmo_t; p++) {
    //    for (size_t q = 0; q < nmo_t; q++) {
    //        size_t pq = p * nmo_t + q;
    //        for (size_t k = 0; k < n_occupied_t; k++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    double a = 0.0;
    //                for (size_t r = 0; r < nmo_t; r++) {
    //    	        size_t rl = r * n_occupied_t + l;
    //                    a += temp10[pq * nmo_t * n_occupied_t + rl] * U[r * nmo_t + k];
    //    	    }	
    //                temp11[pq * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}
    //for (size_t p = 0; p < nmo_t; p++) {
    //    for (size_t s = 0; s < nmo_t; s++) {
    //        size_t ps = p * nmo_t + s;
    //        for (size_t k = 0; k < n_occupied_t; k++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    size_t kl = k * n_occupied_t + l;
    //    	    double a = 0.0;
    //                for (size_t q = 0; q < nmo_t; q++) {
    //    	        size_t pq = p * nmo_t + q;
    //                    a += temp11[pq * n_occupied_t * n_occupied_t + kl] * U[q * nmo_t + s];
    //    	    }	
    //                temp12[ps * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}

    //for (size_t r = 0; r < nmo_t; r++) {
    //    for (size_t s = 0; s < nmo_t; s++) {
    //        size_t rs = r * nmo_t + s;
    //        for (size_t k = 0; k < n_occupied_t; k++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    size_t kl = k * n_occupied_t + l;
    //    	    double a = 0.0;
    //                for (size_t p = 0; p < nmo_t; p++) {
    //    	        size_t ps = p * nmo_t + s;
    //                    a += temp12[ps * n_occupied_t * n_occupied_t + kl] * U[p * nmo_t + r];
    //    	    }	
    //                temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}
    //
    //double* temp14 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp14, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
    //double* temp15 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp15, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
 


 
    //for (size_t k_p = 0; k_p < n_occupied_t; k_p++) {
    //    for (size_t l_p = 0; l_p < n_occupied_t; l_p++) {
    //        for (size_t r_p = 0; r_p < nmo_t; r_p++) {
    //            for (size_t s_p = 0; s_p < nmo_t; s_p++) {
    //                for (size_t p = 0; p < nmo_t; p++) {
    //                    for (size_t q = 0; q < nmo_t; q++) {
    //                        for (size_t r = 0; r < nmo_t; r++) {
    //                            for (size_t s = 0; s < nmo_t; s++) {
    //    			    temp14[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[p * nmo_t + k_p] * U[q * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * h2e[p * nmo_t * nmo_t * nmo_t + q * nmo_t * nmo_t + r * nmo_t +s];	   
    //                                temp15[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[p * nmo_t + k_p] * U[q * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * h2e[r * nmo_t * nmo_t * nmo_t + p * nmo_t * nmo_t + s * nmo_t +q];	  
    //    			}
    //    		    }
    //    		}
    //    	    }
    //    	}
    //        }
    //    }
    //}
    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //                //print("%20.12lf %20.12lf %20.12lf\n", temp14[kl * nmo_t * nmo_t + rs], temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l], J[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J[kl * nmo_t * nmo_t + rs]);
    //    	    //J[kl * nmo_t * nmo_t + rs] = temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l];
    //    	}
    //        }
    //    }
    //}
   



    //build K
    #pragma omp parallel for num_threads(16)
    for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
        //size_t stride = pq_up * 2;
        //size_t p = index_map_pq[stride];
        //size_t q = index_map_pq[stride+1];
	//size_t pq = p * nmo_t + q;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, 1.0, h2e_half+pq_up*nmo_t*nmo_t,
                  nmo_t, U, nmo_t, 0.0,
                  temp1+pq_up*nmo_t*n_occupied_t, n_occupied_t);
    }

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //    	    a += h2e_half[pq_up * nmo_t * nmo_t + rs] * U[s * nmo_t + l];
    //            }
    //    	temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] = a;
    //        }
    //    }
    //}
    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l], temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //            print("%20.12lf \n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] - temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //        }
    //    }
    //}

    for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
        //size_t stride = pq_up * 2;
        //size_t p = index_map_pq[stride];
        //size_t q = index_map_pq[stride+1];
	//size_t pq = p * nmo_t + q;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, nmo_t, 1.0, U,
                  nmo_t, temp1+pq_up*nmo_t*n_occupied_t, n_occupied_t, 0.0,
                  temp3+pq_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t r = 0; r < nmo_t; r++) {
    //                size_t rl = r * n_occupied_t + l;
    //    	    a += temp1[pq_up * nmo_t * n_occupied_t + rl] * U[r * nmo_t + k];
    //            }
    //    	temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //        }
    //    }
    //}

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l], temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //            print("%20.12lf \n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]- temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //        }	
    //    }
    //}

    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    temp5[kl_up * nmo_t * nmo_t + pq] = temp3[pq_up * n_occupied_t * n_occupied_t + kl];
	    temp5[kl_up * nmo_t * nmo_t + qp] = temp3[pq_up * n_occupied_t * n_occupied_t + lk];
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, temp5+kl_up*nmo_t*nmo_t,
                  nmo_t, U, nmo_t, 0.0,
                  temp6+kl_up*nmo_t*nmo_t, nmo_t);
    }
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t q = 0; q < nmo_t; q++) {
    //                size_t pq = p * nmo_t + q;
    //    	    a += temp5[kl_up * nmo_t * nmo_t + pq] * U[q * nmo_t + s];
    //            }
    //    	temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] = a;
    //        }
    //    }
    //}
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s], temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //            print("%20.12lf \n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] - temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //        }
    //    }
    //}
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp6+kl_up*nmo_t*nmo_t, nmo_t, 0.0,
                  temp8+kl_up*nmo_t*nmo_t, nmo_t);
    }
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t p = 0; p < nmo_t; p++) {
    //                size_t ps = p * nmo_t + s;
    //    	    a += temp6[kl_up * nmo_t * nmo_t + ps] * U[p * nmo_t + r];
    //            }
    //    	temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] = a;
    //        }
    //    }
    //}

    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s], temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //            print("%20.12lf \n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] - temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //        }
    //    }
    //}
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    K[kl * nmo_t * nmo_t + pq] = temp8[kl_up * nmo_t * nmo_t + pq];
	    K[lk * nmo_t * nmo_t + pq] = temp8[kl_up * nmo_t * nmo_t + qp];
	    K[kl * nmo_t * nmo_t + qp] = K[lk * nmo_t * nmo_t + pq];
	    K[lk * nmo_t * nmo_t + qp] = K[kl * nmo_t * nmo_t + pq];
	}
    }
    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //                //print("%20.12lf %20.12lf %20.12lf\n", temp14[kl * nmo_t * nmo_t + rs], temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l], J[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf\n", temp15[kl * nmo_t * nmo_t + rs] -  K[kl * nmo_t * nmo_t + rs]);
    //    	}
    //        }
    //    }
    //}
   

    //fflush(stdout);
    free(h2e_half);
    free(temp1);
    //free(temp2);
    free(temp3);
    //free(temp4);
    free(temp5);
    free(temp6);
    //free(temp7);
    free(temp8);
    //free(temp9);
    //free(temp10);
    //free(temp11);
    //free(temp12);
    //free(temp13);
    //free(temp14);
    //free(temp15);
}

void full_transformation_internal_optimization(double* U, double* J, double *K, double* h, double *d_cmo, 
		double* J1, double* K1, double* h1, double *d_cmo1, int* index_map_ab, int* index_map_kl, int nmo, int n_occupied) {
    size_t nmo_t = (size_t) nmo;
    size_t n_occupied_t = (size_t) n_occupied;

    size_t n_virtual_t = nmo_t - n_occupied_t;
    double* J_half = (double*) malloc((size_t)n_virtual_t *(n_virtual_t+1)/2 * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(J_half, 0, n_virtual_t *(n_virtual_t+1)/2 *  n_occupied_t *  n_occupied_t * sizeof(double));
    double* temp1 = (double*) malloc((size_t)n_virtual_t *(n_virtual_t+1)/2 * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp1, 0, n_virtual_t *(n_virtual_t+1)/2 *  n_occupied_t *  n_occupied_t * sizeof(double));
    double* temp2 = (double*) malloc((size_t)n_virtual_t *(n_virtual_t+1)/2 * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp2, 0, n_virtual_t *(n_virtual_t+1)/2 *  n_occupied_t *  n_occupied_t * sizeof(double));

    //for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
    //    size_t stride = ab_up * 2;
    //    size_t a = index_map_ab[stride];
    //    size_t b = index_map_ab[stride+1];
    //    size_t ab = a * n_virtual_t + b;
    //    print("%4d%4d%4d\n", ab_up, a , b);
    //}


    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];
	//size_t pq = p * nmo_t + q;
	//print("%4d%4d%4d\n", pq_up, p , q);
        for (size_t k = 0; k < n_occupied_t; k++) {
            for (size_t l = 0; l < n_occupied_t; l++) {
		size_t kl = k * n_occupied_t + l;
	        J_half[ab_up * n_occupied_t * n_occupied_t + kl] = J[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, J_half+ab_up*n_occupied_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }

    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t, 0.0,
                  temp2+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride1 = ab_up * 2;
        size_t a = index_map_ab[stride1];
        size_t b = index_map_ab[stride1+1];
        for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
            size_t stride2 = kl_up * 2;
            size_t k = index_map_kl[stride2];
            size_t l = index_map_kl[stride2+1];
	    size_t kl = k * n_occupied_t + l;
	    size_t lk = l * n_occupied_t + k;
	    J1[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    J1[lk * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    J1[kl * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    J1[lk * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	}

    }
    double* temp3 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp3, 0, nmo_t * n_occupied_t *  n_occupied_t *  n_occupied_t * sizeof(double));
    double* temp4 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp4, 0, nmo_t * n_occupied_t *  n_occupied_t *  n_occupied_t * sizeof(double));
    #pragma omp parallel for num_threads(16)
    for (size_t p = 0; p < nmo_t; p++) {
        for (size_t m = 0; m < n_occupied_t; m++) {
            size_t pm = p * n_occupied_t + m;
            for (size_t k = 0; k < n_occupied_t; k++) {
                for (size_t l = 0; l < n_occupied_t; l++) {
                    size_t kl = k * n_occupied_t + l;
                    temp3[pm * n_occupied_t * n_occupied_t + kl] = J[kl * nmo_t * nmo_t + p * nmo_t + m];
		}
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t pm = 0; pm < nmo_t * n_occupied_t; pm++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, temp3+pm*n_occupied_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp4+pm*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for num_threads(16)
    for (size_t pm = 0; pm < nmo_t * n_occupied_t; pm++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp4+pm*n_occupied_t*n_occupied_t, n_occupied_t, 0.0,
                  temp3+pm*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    
    double* temp5 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));
    memset(temp5, 0, nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));
    double* temp6 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));
    memset(temp6, 0, nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));   
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        size_t stride2 = kl_up * 2;
        size_t k = index_map_kl[stride2];
        size_t l = index_map_kl[stride2+1];
        size_t kl = k * n_occupied_t + l;
        size_t lk = l * n_occupied_t + k;
        for (size_t pm = 0; pm < nmo_t * n_occupied_t; pm++) {
            temp5[kl_up * nmo_t * n_occupied_t + pm] = temp3[pm * n_occupied_t * n_occupied_t + kl];
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, n_occupied_t, 1.0, temp5+kl_up*nmo_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp6+kl_up*nmo_t*n_occupied_t, n_occupied_t);
    }
    cblas_dcopy( (size_t)nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2,temp6,1,temp5,1);
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp5+kl_up*nmo_t*n_occupied_t, n_occupied_t, 0.0,
                  temp6+kl_up*nmo_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for num_threads(16)
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        size_t stride2 = kl_up * 2;
        size_t k = index_map_kl[stride2];
        size_t l = index_map_kl[stride2+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t p = 0; p < nmo_t; p++) {
            for (size_t m = 0; m < n_occupied_t; m++) {
		size_t lm = l * n_occupied_t + m;
		size_t ml = m * n_occupied_t + l;
		size_t km = k * n_occupied_t + m;
		size_t mk = m * n_occupied_t + k;
	        J1[kl * nmo_t * nmo_t + p * nmo_t + m] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        J1[lk * nmo_t * nmo_t + p * nmo_t + m] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        J1[kl * nmo_t * nmo_t + m * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        J1[lk * nmo_t * nmo_t + m * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[lm * nmo_t * nmo_t + k * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[km * nmo_t * nmo_t + l * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[ml * nmo_t * nmo_t + p * nmo_t + k] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[mk * nmo_t * nmo_t + p * nmo_t + l] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];
	//size_t pq = p * nmo_t + q;
	//print("%4d%4d%4d\n", pq_up, p , q);
        for (size_t k = 0; k < n_occupied_t; k++) {
            for (size_t l = 0; l < n_occupied_t; l++) {
		size_t kl = k * n_occupied_t + l;
	        J_half[ab_up * n_occupied_t * n_occupied_t + kl] = K[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, J_half+ab_up*n_occupied_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }

    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t, 0.0,
                  temp2+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for num_threads(16)
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride1 = ab_up * 2;
        size_t a = index_map_ab[stride1];
        size_t b = index_map_ab[stride1+1];
        for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
            size_t stride2 = kl_up * 2;
            size_t k = index_map_kl[stride2];
            size_t l = index_map_kl[stride2+1];
	    size_t kl = k * n_occupied_t + l;
	    size_t lk = l * n_occupied_t + k;
	    K1[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    K1[lk * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + lk];
	    K1[kl * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + lk];
	    K1[lk * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	}
    }

    //double* temp14 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp14, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
    //double* temp15 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp15, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
 

 
 
    //for (size_t k_p = 0; k_p < n_occupied_t; k_p++) {
    //    for (size_t l_p = 0; l_p < n_occupied_t; l_p++) {
    //        for (size_t r_p = 0; r_p < nmo_t; r_p++) {
    //            for (size_t s_p = 0; s_p < nmo_t; s_p++) {
    //                for (size_t k = 0; k < n_occupied_t; k++) {
    //                    for (size_t l = 0; l < n_occupied_t; l++) {
    //                        for (size_t r = 0; r < nmo_t; r++) {
    //                            for (size_t s = 0; s < nmo_t; s++) {
    //    			    temp14[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[k * nmo_t + k_p] * U[l * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * J[k * n_occupied_t * nmo_t * nmo_t + l * nmo_t * nmo_t + r * nmo_t +s];	   
    //    			    temp15[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[k * nmo_t + k_p] * U[l * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * K[k * n_occupied_t * nmo_t * nmo_t + l * nmo_t * nmo_t + r * nmo_t +s];	   
    //    			}
    //    		    }
    //    		}
    //    	    }
    //    	}
    //        }
    //    }
    //}
    ////for (size_t k = 0; k < n_occupied_t; k++) {
    ////    for (size_t l = 0; l < n_occupied_t; l++) {
    ////        size_t kl = k * n_occupied_t + l;
    ////        for (size_t r = 0; r < nmo_t; r++) {
    ////            for (size_t s = 0; s < nmo_t; s++) {
    ////                size_t rs = r * nmo_t + s;
    ////                //print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J1[kl * nmo_t * nmo_t + rs]);
    ////                print("%20.12lf %20.12lf%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs],  J1[kl * nmo_t * nmo_t + rs], temp14[kl * nmo_t * nmo_t + rs]-  J1[kl * nmo_t * nmo_t + rs]);
    ////    	}
    ////        }
    ////    }
    ////}
    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //                //print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J1[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf %20.12lf%20.12lf\n", temp15[kl * nmo_t * nmo_t + rs],  K1[kl * nmo_t * nmo_t + rs], temp15[kl * nmo_t * nmo_t + rs]-  K1[kl * nmo_t * nmo_t + rs]);
    //    	}
    //        }
    //    }
    //}
    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t a = 0; a < n_virtual_t; a++) {
    //            for (size_t b = 0; b < n_virtual_t; b++) {
    //                size_t ab = (a +n_occupied_t)* nmo_t + b+n_occupied_t;
    //                //print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J1[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf %20.12lf%20.12lf\n", temp14[kl * nmo_t * nmo_t + ab],  J1[kl * nmo_t * nmo_t + ab], temp14[kl * nmo_t * nmo_t + ab]-  J1[kl * nmo_t * nmo_t + ab]);
    //    	}
    //        }
    //    }
    //}
    double* temp7 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
    memset(temp7, 0, nmo_t * nmo_t * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, h,
                  nmo_t, U, nmo_t, 0.0,
                  temp7, nmo_t);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp7, nmo_t, 0.0,
                  h1, nmo_t);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, d_cmo,
                  nmo_t, U, nmo_t, 0.0,
                  temp7, nmo_t);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp7, nmo_t, 0.0,
                  d_cmo1, nmo_t);
    //fflush(stdout);
    free(temp1);  
    free(temp2);  
    free(temp3);  
    free(temp4);  
    free(temp5);  
    free(temp6);  
    free(temp7);  
    //free(temp14);  
    //free(temp15); 

}




void build_sigma_reduced(double* U, double* A_tilde, int* index_map, double* G, double* R_reduced, double* sigma_reduced, int    num_states, int    pointer, int    nmo, int    index_map_size, int    n_occupied){
    size_t nmo_t = (size_t) nmo;
    size_t n_occupied_t = (size_t) n_occupied;

    double* R_total = (double*) malloc(num_states * nmo_t * n_occupied_t * sizeof(double));
    memset(R_total, 0, num_states * nmo_t * n_occupied_t * sizeof(double));
    double* sigma_total = (double*) malloc(num_states * nmo_t * n_occupied_t * sizeof(double));
    memset(sigma_total, 0, num_states * nmo_t * n_occupied_t * sizeof(double));
    //print("num state %d\n",num_states); 
    //print("index_map_size %d\n",index_map_size); 
    double* A3 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
    memset(A3, 0, nmo_t * nmo_t * sizeof(double));
    #pragma omp parallel for num_threads(16)
    for (size_t p = 0; p < nmo_t; p++) {
        for (size_t q = 0; q < nmo_t; q++) {
	    A3[p * nmo_t + q] = A_tilde[p * nmo_t + q] + A_tilde[q * nmo_t + p];
	}
    }
    
    #pragma omp parallel for num_threads(16)
    for (size_t j = 0; j < index_map_size; j++) {
        size_t r = index_map[j * 2 + 0]; 
        size_t k = index_map[j * 2 + 1];
        //print("%4d  %4d %4d\n",j,r,k);	
        for (size_t i = 0; i < num_states; i++) {
            R_total[i * nmo_t * n_occupied_t + r * n_occupied_t + k] = R_reduced[i * (index_map_size + pointer) + j+pointer];
        }
    } 
     
    
    #pragma omp parallel for num_threads(16)
    for (size_t i = 0; i < num_states; i++) {
        double* R = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        memset(R, 0, nmo_t * n_occupied_t * sizeof(double));
        double* sigma = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        memset(sigma, 0, nmo_t * n_occupied_t * sizeof(double));
        double* temp1 = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        memset(temp1, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp2  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp2, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp3  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp3, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp4 = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp4 , 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp5  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp5, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp6  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp6, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp7  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp7, 0, nmo_t * n_occupied_t * sizeof(double));
	//double* sigma2= (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(sigma2, 0, nmo_t * n_occupied_t * sizeof(double));



        cblas_dcopy(nmo_t * n_occupied_t, R_total + i*nmo_t*n_occupied_t,1,R,1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, 1.0, U,
                 nmo_t, R, n_occupied_t, 0.0,
                 temp1, n_occupied_t);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo_t, n_occupied_t, n_occupied_t, -1.0, U,
                 nmo_t, R, n_occupied_t, 1.0,
                 temp1, n_occupied_t);
        
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
	//	double a = 0.0;    
        //        for (size_t s = 0; s < nmo_t; s++) {
        //            a += U[q * nmo_t + s] * R[s * n_occupied_t +l]; 
	//	}
        //            temp2[q * n_occupied_t + l] = a; 
	//    }
	//}	
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t s = 0; s < n_occupied_t; s++) {
	//	double a = 0.0;    
        //        for (size_t l = 0; l < n_occupied_t; l++) {
        //            a += U[q * nmo_t + l] * R[s * n_occupied_t +l]; 
	//	}
        //            temp2[q * n_occupied_t + s] -= a; 
	//    }
	//}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", temp2[q * n_occupied_t + l],  temp1[q * n_occupied_t + l], temp2[q * n_occupied_t + l]-  temp1[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        //for (size_t p = 0; p < nmo_t; p++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	size_t pk = p * n_occupied_t +k;
	//	double a = 0.0;    
        //        for (size_t q = 0; q < nmo_t; q++) {
        //            for (size_t l = 0; l < n_occupied_t; l++) {
	//	        size_t ql = q * n_occupied_t +l;
        //                a += temp1[q * n_occupied_t + l] * G[ql * nmo_t * n_occupied_t +pk]; 
	//	    }
	//	}
        //            temp3[p * n_occupied_t + k] = a; 
	//    }
	//}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, nmo_t * n_occupied_t, nmo_t * n_occupied_t, 1.0, temp1,
                 nmo_t * n_occupied_t, G, nmo_t * n_occupied_t, 0.0,
                 sigma, nmo_t * n_occupied_t);
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", temp3[q * n_occupied_t + l],  sigma[q * n_occupied_t + l], temp3[q * n_occupied_t + l]-  sigma[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, 1.0, U,
                 nmo_t, sigma, n_occupied_t, 0.0,
                 temp1, n_occupied_t);
	cblas_dcopy(nmo_t * n_occupied_t, temp1,1,sigma,1);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t p = 0; p < nmo_t; p++) {
        //            a += temp3[p * n_occupied_t + k] * U[p * nmo_t +r]; 
	//	}
        //            temp4[r * n_occupied_t + k] = a; 
	//    }
	//}
	//for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] = temp4[r * n_occupied_t + k]; 
	//    }
	//}
        //for (size_t r = 0; r < n_occupied_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] -= temp4[k * n_occupied_t + r]; 
	//    }
	//}
	for (size_t r = 0; r < n_occupied_t; r++) {
            for (size_t k = 0; k < n_occupied_t; k++) {
                sigma[r * n_occupied_t + k] -= temp1[k * n_occupied_t + r]; 
	    }
	}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied_t + l],  sigma2[q * n_occupied_t + l], sigma[q * n_occupied_t + l]-  sigma2[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, -0.5, A3,
                 nmo_t, R, n_occupied_t, 0.0,
                 temp1, n_occupied_t);
        cblas_daxpy(nmo_t * n_occupied_t, 1.0, temp1, 1, sigma, 1);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t s = 0; s < nmo_t; s++) {
        //            a += R[s * n_occupied_t + k] * A3[r * nmo_t +s]; 
	//	}
        //            temp5[r * n_occupied_t + k] = a; 
	//    }
	//}
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] -= 0.5 * temp5[r * n_occupied_t + k]; 
	//    }
	//}
        //for (size_t r = 0; r < n_occupied_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] += 0.5 * temp5[k * n_occupied_t + r]; 
	//    }
	//}
        

	for (size_t r = 0; r < n_occupied_t; r++) {
            for (size_t k = 0; k < n_occupied_t; k++) {
                sigma[r * n_occupied_t + k] -= temp1[k * n_occupied_t + r]; 
	    }
	}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied_t + l],  sigma2[q * n_occupied_t + l], sigma[q * n_occupied_t + l]-  sigma2[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo_t, n_occupied_t, n_occupied_t, 0.5, A3,
                 nmo_t, R, n_occupied_t, 1.0,
                 sigma, n_occupied_t);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t l = 0; l < n_occupied_t; l++) {
        //            a += R[k * n_occupied_t + l] * A3[r * nmo_t +l]; 
	//	}
        //            sigma2[r * n_occupied_t + k] += 0.5 * a; 
	//    }
	//}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo_t, n_occupied_t, n_occupied_t, -0.5, R,
                 n_occupied_t, A3, nmo_t, 1.0,
                 sigma, n_occupied_t);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t l = 0; l < n_occupied_t; l++) {
        //            a += R[r * n_occupied_t + l] * A3[k * nmo_t +l]; 
	//	}
        //            sigma2[r * n_occupied_t + k] -= 0.5 * a; 
	//    }
	//}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied_t + l],  sigma2[q * n_occupied_t + l], sigma[q * n_occupied_t + l]-  sigma2[q * n_occupied_t + l]);
	//	    
	//    }
	//}
	cblas_dcopy(nmo_t * n_occupied_t, sigma,1,sigma_total + i*nmo_t*n_occupied_t,1);
        free(R);
        free(sigma);
        free(temp1);
        //free(temp2);
        //free(temp3);
        //free(temp4);
        //free(temp5);
        //free(temp6);
        //free(temp7);
        //free(sigma2);
    } 
   
    #pragma omp parallel for num_threads(16)
    for (size_t j = 0; j < index_map_size; j++) {
        size_t r = index_map[j * 2 + 0]; 
        size_t k = index_map[j * 2 + 1]; 
        for (size_t i = 0; i < num_states; i++) {
            sigma_reduced[i * (index_map_size + pointer) + j+pointer] = sigma_total[i * nmo_t * n_occupied_t + r * n_occupied_t + k];
        }
    } 
     
free(sigma_total);
free(R_total);
free(A3);


}


