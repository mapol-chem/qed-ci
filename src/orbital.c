#include <stdlib.h>
#include <stdio.h>
#include "orbital.h"
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include<omp.h>
#include<time.h>
#include <unistd.h>


void full_transformation_macroiteration(double* U, double* h2e, double* J, double *K, int* index_map_pq, int* index_map_kl, int nmo, int n_occupied) {
    double* h2e_half = (double*) malloc(nmo *(nmo+1)/2 * nmo * nmo * sizeof(double));
    memset(h2e_half, 0, nmo *(nmo+1)/2 * nmo * nmo * sizeof(double));
    double* temp1 = (double*) malloc(nmo *(nmo+1)/2 * nmo * n_occupied * sizeof(double));
    memset(temp1, 0, nmo *(nmo+1)/2 * nmo * n_occupied * sizeof(double));
    double* temp2 = (double*) malloc(nmo *(nmo+1)/2 * nmo * n_occupied * sizeof(double));
    memset(temp2, 0, nmo *(nmo+1)/2 * nmo * n_occupied * sizeof(double));
    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    int stride = pq_up * 2;
    //    int p = index_map_pq[stride];
    //    int q = index_map_pq[stride+1];
    //    int pq = p * nmo + q;
    //    printf("%4d%4d%4d\n", pq_up, p , q);
    //}


    #pragma omp parallel for num_threads(16)
    for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
	int stride = pq_up * 2;
        int p = index_map_pq[stride];
        int q = index_map_pq[stride+1];
	//int pq = p * nmo + q;
	//printf("%4d%4d%4d\n", pq_up, p , q);
        for (int r = 0; r < nmo; r++) {
            for (int s = 0; s < nmo; s++) {
		int rs = r * nmo + s;
		int pr = p * nmo + r;
		int qs = q * nmo + s;
	        h2e_half[pq_up * nmo * nmo + rs] = h2e[pr * nmo * nmo + qs];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
        int stride = pq_up * 2;
        int p = index_map_pq[stride];
        int q = index_map_pq[stride+1];
	int pq = p * nmo + q;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, n_occupied, nmo, 1.0, h2e+pq*nmo*nmo,
                  nmo, U, nmo, 0.0,
                  temp1+pq_up*nmo*n_occupied, n_occupied);
    }

    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    int stride = pq_up * 2;
    //    int p = index_map_pq[stride];
    //    int q = index_map_pq[stride+1];
    //    int pq = p * nmo + q;
    //    for (int r = 0; r < nmo; r++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //    	double a = 0.0;
    //            for (int s = 0; s < nmo; s++) {
    //                int rs = r * nmo + s;
    //    	    a += h2e[pq * nmo * nmo + rs] * U[s * nmo + l];
    //            }
    //    	temp2[pq_up * nmo * n_occupied + r * n_occupied + l] = a;
    //        }
    //    }
    //}
    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    for (int r = 0; r < nmo; r++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //            //printf("%20.12lf %20.12lf\n", temp2[pq_up * nmo * n_occupied + r * n_occupied + l], temp1[pq_up * nmo * n_occupied + r * n_occupied + l]);
    //            printf("%20.12lf \n", temp2[pq_up * nmo * n_occupied + r * n_occupied + l] - temp1[pq_up * nmo * n_occupied + r * n_occupied + l]);
    //        }
    //    }
    //}


    double* temp3 = (double*) malloc(nmo *(nmo+1)/2 * n_occupied * n_occupied * sizeof(double));
    memset(temp3, 0, nmo *(nmo+1)/2 * n_occupied * n_occupied * sizeof(double));
    //double* temp4 = (double*) malloc(nmo *(nmo+1)/2 * n_occupied * n_occupied * sizeof(double));
    //memset(temp4, 0, nmo *(nmo+1)/2 * n_occupied * n_occupied * sizeof(double));
    // when matrix is transposed, in cblas_dgemm, m,n,k are the physical size of transposed matrix
    // but lda,ldb are leading dimensions (strides to next row) of original matrices (number of columns)
    #pragma omp parallel for num_threads(16)
    for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
        //int stride = pq_up * 2;
        //int p = index_map_pq[stride];
        //int q = index_map_pq[stride+1];
	//int pq = p * nmo + q;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied, n_occupied, nmo, 1.0, U,
                  nmo, temp1+pq_up*nmo*n_occupied, n_occupied, 0.0,
                  temp3+pq_up*n_occupied*n_occupied, n_occupied);
    }

    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    int stride = pq_up * 2;
    //    int p = index_map_pq[stride];
    //    int q = index_map_pq[stride+1];
    //    int pq = p * nmo + q;
    //    for (int k = 0; k < n_occupied; k++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //    	double a = 0.0;
    //            for (int r = 0; r < nmo; r++) {
    //                int rl = r * n_occupied + l;
    //    	    a += temp1[pq_up * nmo * n_occupied + rl] * U[r * nmo + k];
    //            }
    //    	temp4[pq_up * n_occupied * n_occupied + k * n_occupied + l] = a;
    //        }
    //    }
    //}

    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    for (int k = 0; k < n_occupied; k++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //            //printf("%20.12lf %20.12lf\n", temp4[pq_up * n_occupied * n_occupied + k * n_occupied + l], temp3[pq_up * n_occupied * n_occupied + k * n_occupied + l]);
    //            printf("%20.12lf \n", temp4[pq_up * n_occupied * n_occupied + k * n_occupied + l]- temp3[pq_up * n_occupied * n_occupied + k * n_occupied + l]);
    //        }	
    //    }
    //}
    double* temp5 = (double*) malloc(n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    memset(temp5, 0, n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    double* temp6 = (double*) malloc(n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    memset(temp6, 0, n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	int stride = kl_up * 2;
        int k = index_map_kl[stride];
        int l = index_map_kl[stride+1];
	int kl = k * n_occupied + l;
        for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
            int stride1 = pq_up * 2;
            int p = index_map_pq[stride1];
            int q = index_map_pq[stride1+1];
	    int pq = p * nmo + q;
	    int qp = q * nmo + p;
	    temp5[kl_up * nmo * nmo + pq] = temp3[pq_up * n_occupied * n_occupied + kl];
	    temp5[kl_up * nmo * nmo + qp] = temp5[kl_up * nmo * nmo + pq];
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	//int stride = kl_up * 2;
        //int k = index_map_kl[stride];
        //int l = index_map_kl[stride+1];
	//int kl = k * n_occupied + l;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, temp5+kl_up*nmo*nmo,
                  nmo, U, nmo, 0.0,
                  temp6+kl_up*nmo*nmo, nmo);
    }
    //double* temp7 = (double*) malloc(n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    //memset(temp7, 0, n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int p = 0; p < nmo; p++) {
    //        for (int s = 0; s < nmo; s++) {
    //    	double a = 0.0;
    //            for (int q = 0; q < nmo; q++) {
    //                int pq = p * nmo + q;
    //    	    a += temp5[kl_up * nmo * nmo + pq] * U[q * nmo + s];
    //            }
    //    	temp7[kl_up * nmo * nmo + p * nmo + s] = a;
    //        }
    //    }
    //}
    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int p = 0; p < nmo; p++) {
    //        for (int s = 0; s < nmo; s++) {
    //            //printf("%20.12lf %20.12lf\n", temp7[kl_up * nmo * nmo + p * nmo + s], temp6[kl_up * nmo * nmo + p * nmo + s]);
    //            printf("%20.12lf \n", temp7[kl_up * nmo * nmo + p * nmo + s] - temp6[kl_up * nmo * nmo + p * nmo + s]);
    //        }
    //    }
    //}
    double* temp8 = (double*) malloc(n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    memset(temp8, 0, n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    //double* temp9 = (double*) malloc(n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    //memset(temp9, 0, n_occupied *(n_occupied+1)/2 * nmo * nmo * sizeof(double));
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	//int stride = kl_up * 2;
        //int k = index_map_kl[stride];
        //int l = index_map_kl[stride+1];
	//int kl = k * n_occupied + l;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, U,
                  nmo, temp6+kl_up*nmo*nmo, nmo, 0.0,
                  temp8+kl_up*nmo*nmo, nmo);
    }
    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int r = 0; r < nmo; r++) {
    //        for (int s = 0; s < nmo; s++) {
    //    	double a = 0.0;
    //            for (int p = 0; p < nmo; p++) {
    //                int ps = p * nmo + s;
    //    	    a += temp6[kl_up * nmo * nmo + ps] * U[p * nmo + r];
    //            }
    //    	temp9[kl_up * nmo * nmo + r * nmo + s] = a;
    //        }
    //    }
    //}
    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int r = 0; r < nmo; r++) {
    //        for (int s = 0; s < nmo; s++) {
    //            //printf("%20.12lf %20.12lf\n", temp9[kl_up * nmo * nmo + r * nmo + s], temp8[kl_up * nmo * nmo + r * nmo + s]);
    //            printf("%20.12lf \n", temp9[kl_up * nmo * nmo + r * nmo + s] - temp8[kl_up * nmo * nmo + r * nmo + s]);
    //        }
    //    }
    //}
    //double* temp10 = (double*) malloc(nmo * nmo * nmo * n_occupied* sizeof(double));
    //memset(temp10, 0,  nmo * nmo * nmo * n_occupied * sizeof(double));
    //double* temp11 = (double*) malloc( nmo * nmo * n_occupied * n_occupied* sizeof(double));
    //memset(temp11, 0, nmo * nmo * n_occupied * n_occupied * sizeof(double));
    //double* temp12 = (double*) malloc( nmo * nmo * n_occupied * n_occupied* sizeof(double));
    //memset(temp12, 0, nmo * nmo * n_occupied * n_occupied * sizeof(double));
    //double* temp13 = (double*) malloc( nmo * nmo * n_occupied * n_occupied* sizeof(double));
    //memset(temp13, 0, nmo * nmo * n_occupied * n_occupied * sizeof(double));
    //fflush(stdout);
 

    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	int stride = kl_up * 2;
        int k = index_map_kl[stride];
        int l = index_map_kl[stride+1];
	int kl = k * n_occupied + l;
	int lk = l * n_occupied + k;
        for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
            int stride1 = pq_up * 2;
            int p = index_map_pq[stride1];
            int q = index_map_pq[stride1+1];
	    int pq = p * nmo + q;
	    int qp = q * nmo + p;
	    J[kl * nmo * nmo + pq] = temp8[kl_up * nmo * nmo + pq];
	    J[kl * nmo * nmo + qp] = J[kl * nmo * nmo + pq];
	    J[lk * nmo * nmo + pq] = J[kl * nmo * nmo + pq];
	    J[lk * nmo * nmo + qp] = J[kl * nmo * nmo + pq];
	}
    }
    //printf("test full transformation\n");
    //for (int p = 0; p < nmo; p++) {
    //    for (int q = 0; q < nmo; q++) {
    //        int pq = p * nmo + q;
    //        for (int r = 0; r < nmo; r++) {
    //            for (int l = 0; l < n_occupied; l++) {
    //    	    double a = 0.0;
    //                for (int s = 0; s < nmo; s++) {
    //    	        int rs = r * nmo + s;
    //                    a += h2e[pq * nmo * nmo + rs] * U[s * nmo + l];
    //    	    }	
    //                temp10[pq * nmo * n_occupied + r * n_occupied + l] = a;
    //    	}
    //        }
    //    }
    //}
    //for (int p = 0; p < nmo; p++) {
    //    for (int q = 0; q < nmo; q++) {
    //        int pq = p * nmo + q;
    //        for (int k = 0; k < n_occupied; k++) {
    //            for (int l = 0; l < n_occupied; l++) {
    //    	    double a = 0.0;
    //                for (int r = 0; r < nmo; r++) {
    //    	        int rl = r * n_occupied + l;
    //                    a += temp10[pq * nmo * n_occupied + rl] * U[r * nmo + k];
    //    	    }	
    //                temp11[pq * n_occupied * n_occupied + k * n_occupied + l] = a;
    //    	}
    //        }
    //    }
    //}
    //for (int p = 0; p < nmo; p++) {
    //    for (int s = 0; s < nmo; s++) {
    //        int ps = p * nmo + s;
    //        for (int k = 0; k < n_occupied; k++) {
    //            for (int l = 0; l < n_occupied; l++) {
    //    	    int kl = k * n_occupied + l;
    //    	    double a = 0.0;
    //                for (int q = 0; q < nmo; q++) {
    //    	        int pq = p * nmo + q;
    //                    a += temp11[pq * n_occupied * n_occupied + kl] * U[q * nmo + s];
    //    	    }	
    //                temp12[ps * n_occupied * n_occupied + k * n_occupied + l] = a;
    //    	}
    //        }
    //    }
    //}

    //for (int r = 0; r < nmo; r++) {
    //    for (int s = 0; s < nmo; s++) {
    //        int rs = r * nmo + s;
    //        for (int k = 0; k < n_occupied; k++) {
    //            for (int l = 0; l < n_occupied; l++) {
    //    	    int kl = k * n_occupied + l;
    //    	    double a = 0.0;
    //                for (int p = 0; p < nmo; p++) {
    //    	        int ps = p * nmo + s;
    //                    a += temp12[ps * n_occupied * n_occupied + kl] * U[p * nmo + r];
    //    	    }	
    //                temp13[rs * n_occupied * n_occupied + k * n_occupied + l] = a;
    //    	}
    //        }
    //    }
    //}
    //
    //double* temp14 = (double*) malloc(n_occupied * n_occupied * nmo * nmo * sizeof(double));
    //memset(temp14, 0,n_occupied * n_occupied *  nmo * nmo * sizeof(double));
    //double* temp15 = (double*) malloc(n_occupied * n_occupied * nmo * nmo * sizeof(double));
    //memset(temp15, 0,n_occupied * n_occupied *  nmo * nmo * sizeof(double));
 


 
    //for (int k_p = 0; k_p < n_occupied; k_p++) {
    //    for (int l_p = 0; l_p < n_occupied; l_p++) {
    //        for (int r_p = 0; r_p < nmo; r_p++) {
    //            for (int s_p = 0; s_p < nmo; s_p++) {
    //                for (int p = 0; p < nmo; p++) {
    //                    for (int q = 0; q < nmo; q++) {
    //                        for (int r = 0; r < nmo; r++) {
    //                            for (int s = 0; s < nmo; s++) {
    //    			    temp14[k_p * n_occupied * nmo * nmo +l_p *nmo * nmo + r_p * nmo + s_p] +=
    //    			    U[p * nmo + k_p] * U[q * nmo + l_p] * U[r * nmo + r_p] * U[s * nmo + s_p] * h2e[p * nmo * nmo * nmo + q * nmo * nmo + r * nmo +s];	   
    //                                temp15[k_p * n_occupied * nmo * nmo +l_p *nmo * nmo + r_p * nmo + s_p] +=
    //    			    U[p * nmo + k_p] * U[q * nmo + l_p] * U[r * nmo + r_p] * U[s * nmo + s_p] * h2e[r * nmo * nmo * nmo + p * nmo * nmo + s * nmo +q];	  
    //    			}
    //    		    }
    //    		}
    //    	    }
    //    	}
    //        }
    //    }
    //}
    //for (int k = 0; k < n_occupied; k++) {
    //    for (int l = 0; l < n_occupied; l++) {
    //        int kl = k * n_occupied + l;
    //        for (int r = 0; r < nmo; r++) {
    //            for (int s = 0; s < nmo; s++) {
    //                int rs = r * nmo + s;
    //                //printf("%20.12lf %20.12lf %20.12lf\n", temp14[kl * nmo * nmo + rs], temp13[rs * n_occupied * n_occupied + k * n_occupied + l], J[kl * nmo * nmo + rs]);
    //                printf("%20.12lf\n", temp14[kl * nmo * nmo + rs] -  J[kl * nmo * nmo + rs]);
    //    	    //J[kl * nmo * nmo + rs] = temp13[rs * n_occupied * n_occupied + k * n_occupied + l];
    //    	}
    //        }
    //    }
    //}
   



    //build K
    #pragma omp parallel for num_threads(16)
    for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
        //int stride = pq_up * 2;
        //int p = index_map_pq[stride];
        //int q = index_map_pq[stride+1];
	//int pq = p * nmo + q;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, n_occupied, nmo, 1.0, h2e_half+pq_up*nmo*nmo,
                  nmo, U, nmo, 0.0,
                  temp1+pq_up*nmo*n_occupied, n_occupied);
    }

    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    int stride = pq_up * 2;
    //    int p = index_map_pq[stride];
    //    int q = index_map_pq[stride+1];
    //    int pq = p * nmo + q;
    //    for (int r = 0; r < nmo; r++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //    	double a = 0.0;
    //            for (int s = 0; s < nmo; s++) {
    //                int rs = r * nmo + s;
    //    	    a += h2e_half[pq_up * nmo * nmo + rs] * U[s * nmo + l];
    //            }
    //    	temp2[pq_up * nmo * n_occupied + r * n_occupied + l] = a;
    //        }
    //    }
    //}
    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    for (int r = 0; r < nmo; r++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //            //printf("%20.12lf %20.12lf\n", temp2[pq_up * nmo * n_occupied + r * n_occupied + l], temp1[pq_up * nmo * n_occupied + r * n_occupied + l]);
    //            printf("%20.12lf \n", temp2[pq_up * nmo * n_occupied + r * n_occupied + l] - temp1[pq_up * nmo * n_occupied + r * n_occupied + l]);
    //        }
    //    }
    //}

    for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
        //int stride = pq_up * 2;
        //int p = index_map_pq[stride];
        //int q = index_map_pq[stride+1];
	//int pq = p * nmo + q;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied, n_occupied, nmo, 1.0, U,
                  nmo, temp1+pq_up*nmo*n_occupied, n_occupied, 0.0,
                  temp3+pq_up*n_occupied*n_occupied, n_occupied);
    }

    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    int stride = pq_up * 2;
    //    int p = index_map_pq[stride];
    //    int q = index_map_pq[stride+1];
    //    int pq = p * nmo + q;
    //    for (int k = 0; k < n_occupied; k++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //    	double a = 0.0;
    //            for (int r = 0; r < nmo; r++) {
    //                int rl = r * n_occupied + l;
    //    	    a += temp1[pq_up * nmo * n_occupied + rl] * U[r * nmo + k];
    //            }
    //    	temp4[pq_up * n_occupied * n_occupied + k * n_occupied + l] = a;
    //        }
    //    }
    //}

    //for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
    //    for (int k = 0; k < n_occupied; k++) {
    //        for (int l = 0; l < n_occupied; l++) {
    //            //printf("%20.12lf %20.12lf\n", temp4[pq_up * n_occupied * n_occupied + k * n_occupied + l], temp3[pq_up * n_occupied * n_occupied + k * n_occupied + l]);
    //            printf("%20.12lf \n", temp4[pq_up * n_occupied * n_occupied + k * n_occupied + l]- temp3[pq_up * n_occupied * n_occupied + k * n_occupied + l]);
    //        }	
    //    }
    //}

    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	int stride = kl_up * 2;
        int k = index_map_kl[stride];
        int l = index_map_kl[stride+1];
	int kl = k * n_occupied + l;
	int lk = l * n_occupied + k;
        for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
            int stride1 = pq_up * 2;
            int p = index_map_pq[stride1];
            int q = index_map_pq[stride1+1];
	    int pq = p * nmo + q;
	    int qp = q * nmo + p;
	    temp5[kl_up * nmo * nmo + pq] = temp3[pq_up * n_occupied * n_occupied + kl];
	    temp5[kl_up * nmo * nmo + qp] = temp3[pq_up * n_occupied * n_occupied + lk];
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	//int stride = kl_up * 2;
        //int k = index_map_kl[stride];
        //int l = index_map_kl[stride+1];
	//int kl = k * n_occupied + l;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, temp5+kl_up*nmo*nmo,
                  nmo, U, nmo, 0.0,
                  temp6+kl_up*nmo*nmo, nmo);
    }
    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int p = 0; p < nmo; p++) {
    //        for (int s = 0; s < nmo; s++) {
    //    	double a = 0.0;
    //            for (int q = 0; q < nmo; q++) {
    //                int pq = p * nmo + q;
    //    	    a += temp5[kl_up * nmo * nmo + pq] * U[q * nmo + s];
    //            }
    //    	temp7[kl_up * nmo * nmo + p * nmo + s] = a;
    //        }
    //    }
    //}
    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int p = 0; p < nmo; p++) {
    //        for (int s = 0; s < nmo; s++) {
    //            //printf("%20.12lf %20.12lf\n", temp7[kl_up * nmo * nmo + p * nmo + s], temp6[kl_up * nmo * nmo + p * nmo + s]);
    //            printf("%20.12lf \n", temp7[kl_up * nmo * nmo + p * nmo + s] - temp6[kl_up * nmo * nmo + p * nmo + s]);
    //        }
    //    }
    //}
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	//int stride = kl_up * 2;
        //int k = index_map_kl[stride];
        //int l = index_map_kl[stride+1];
	//int kl = k * n_occupied + l;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, U,
                  nmo, temp6+kl_up*nmo*nmo, nmo, 0.0,
                  temp8+kl_up*nmo*nmo, nmo);
    }
    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int r = 0; r < nmo; r++) {
    //        for (int s = 0; s < nmo; s++) {
    //    	double a = 0.0;
    //            for (int p = 0; p < nmo; p++) {
    //                int ps = p * nmo + s;
    //    	    a += temp6[kl_up * nmo * nmo + ps] * U[p * nmo + r];
    //            }
    //    	temp9[kl_up * nmo * nmo + r * nmo + s] = a;
    //        }
    //    }
    //}

    //for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
    //    int stride = kl_up * 2;
    //    int k = index_map_kl[stride];
    //    int l = index_map_kl[stride+1];
    //    int kl = k * n_occupied + l;
    //    for (int r = 0; r < nmo; r++) {
    //        for (int s = 0; s < nmo; s++) {
    //            //printf("%20.12lf %20.12lf\n", temp9[kl_up * nmo * nmo + r * nmo + s], temp8[kl_up * nmo * nmo + r * nmo + s]);
    //            printf("%20.12lf \n", temp9[kl_up * nmo * nmo + r * nmo + s] - temp8[kl_up * nmo * nmo + r * nmo + s]);
    //        }
    //    }
    //}
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
	int stride = kl_up * 2;
        int k = index_map_kl[stride];
        int l = index_map_kl[stride+1];
	int kl = k * n_occupied + l;
	int lk = l * n_occupied + k;
        for (int pq_up = 0; pq_up < nmo*(nmo+1)/2; pq_up++) {
            int stride1 = pq_up * 2;
            int p = index_map_pq[stride1];
            int q = index_map_pq[stride1+1];
	    int pq = p * nmo + q;
	    int qp = q * nmo + p;
	    K[kl * nmo * nmo + pq] = temp8[kl_up * nmo * nmo + pq];
	    K[lk * nmo * nmo + pq] = temp8[kl_up * nmo * nmo + qp];
	    K[kl * nmo * nmo + qp] = K[lk * nmo * nmo + pq];
	    K[lk * nmo * nmo + qp] = K[kl * nmo * nmo + pq];
	}
    }
    //for (int k = 0; k < n_occupied; k++) {
    //    for (int l = 0; l < n_occupied; l++) {
    //        int kl = k * n_occupied + l;
    //        for (int r = 0; r < nmo; r++) {
    //            for (int s = 0; s < nmo; s++) {
    //                int rs = r * nmo + s;
    //                //printf("%20.12lf %20.12lf %20.12lf\n", temp14[kl * nmo * nmo + rs], temp13[rs * n_occupied * n_occupied + k * n_occupied + l], J[kl * nmo * nmo + rs]);
    //                printf("%20.12lf\n", temp15[kl * nmo * nmo + rs] -  K[kl * nmo * nmo + rs]);
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

    int n_virtual = nmo - n_occupied;
    double* J_half = (double*) malloc(n_virtual *(n_virtual+1)/2 * n_occupied *  n_occupied * sizeof(double));
    memset(J_half, 0, n_virtual *(n_virtual+1)/2 *  n_occupied *  n_occupied * sizeof(double));
    double* temp1 = (double*) malloc(n_virtual *(n_virtual+1)/2 * n_occupied *  n_occupied * sizeof(double));
    memset(temp1, 0, n_virtual *(n_virtual+1)/2 *  n_occupied *  n_occupied * sizeof(double));
    double* temp2 = (double*) malloc(n_virtual *(n_virtual+1)/2 * n_occupied *  n_occupied * sizeof(double));
    memset(temp2, 0, n_virtual *(n_virtual+1)/2 *  n_occupied *  n_occupied * sizeof(double));

    //for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
    //    int stride = ab_up * 2;
    //    int a = index_map_ab[stride];
    //    int b = index_map_ab[stride+1];
    //    int ab = a * n_virtual + b;
    //    printf("%4d%4d%4d\n", ab_up, a , b);
    //}


    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride = ab_up * 2;
        int a = index_map_ab[stride];
        int b = index_map_ab[stride+1];
	//int pq = p * nmo + q;
	//printf("%4d%4d%4d\n", pq_up, p , q);
        for (int k = 0; k < n_occupied; k++) {
            for (int l = 0; l < n_occupied; l++) {
		int kl = k * n_occupied + l;
	        J_half[ab_up * n_occupied * n_occupied + kl] = J[kl * nmo * nmo + (a + n_occupied) * nmo + b + n_occupied];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride = ab_up * 2;
        int a = index_map_ab[stride];
        int b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied, n_occupied, n_occupied, 1.0, J_half+ab_up*n_occupied*n_occupied,
                  n_occupied, U, nmo, 0.0,
                  temp1+ab_up*n_occupied*n_occupied, n_occupied);
    }

    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride = ab_up * 2;
        int a = index_map_ab[stride];
        int b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied, n_occupied, n_occupied, 1.0, U,
                  nmo, temp1+ab_up*n_occupied*n_occupied, n_occupied, 0.0,
                  temp2+ab_up*n_occupied*n_occupied, n_occupied);
    }
    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride1 = ab_up * 2;
        int a = index_map_ab[stride1];
        int b = index_map_ab[stride1+1];
        for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
            int stride2 = kl_up * 2;
            int k = index_map_kl[stride2];
            int l = index_map_kl[stride2+1];
	    int kl = k * n_occupied + l;
	    int lk = l * n_occupied + k;
	    J1[kl * nmo * nmo + (a + n_occupied) * nmo + b + n_occupied] = temp2[ab_up * n_occupied * n_occupied + kl];
	    J1[lk * nmo * nmo + (a + n_occupied) * nmo + b + n_occupied] = temp2[ab_up * n_occupied * n_occupied + kl];
	    J1[kl * nmo * nmo + (b + n_occupied) * nmo + a + n_occupied] = temp2[ab_up * n_occupied * n_occupied + kl];
	    J1[lk * nmo * nmo + (b + n_occupied) * nmo + a + n_occupied] = temp2[ab_up * n_occupied * n_occupied + kl];
	}

    }
    double* temp3 = (double*) malloc(nmo * n_occupied * n_occupied *  n_occupied * sizeof(double));
    memset(temp3, 0, nmo * n_occupied *  n_occupied *  n_occupied * sizeof(double));
    double* temp4 = (double*) malloc(nmo * n_occupied * n_occupied *  n_occupied * sizeof(double));
    memset(temp4, 0, nmo * n_occupied *  n_occupied *  n_occupied * sizeof(double));
    #pragma omp parallel for num_threads(16)
    for (int p = 0; p < nmo; p++) {
        for (int m = 0; m < n_occupied; m++) {
            int pm = p * n_occupied + m;
            for (int k = 0; k < n_occupied; k++) {
                for (int l = 0; l < n_occupied; l++) {
                    int kl = k * n_occupied + l;
                    temp3[pm * n_occupied * n_occupied + kl] = J[kl * nmo * nmo + p * nmo + m];
		}
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int pm = 0; pm < nmo * n_occupied; pm++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied, n_occupied, n_occupied, 1.0, temp3+pm*n_occupied*n_occupied,
                  n_occupied, U, nmo, 0.0,
                  temp4+pm*n_occupied*n_occupied, n_occupied);
    }
    #pragma omp parallel for num_threads(16)
    for (int pm = 0; pm < nmo * n_occupied; pm++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied, n_occupied, n_occupied, 1.0, U,
                  nmo, temp4+pm*n_occupied*n_occupied, n_occupied, 0.0,
                  temp3+pm*n_occupied*n_occupied, n_occupied);
    }
    
    double* temp5 = (double*) malloc(nmo * n_occupied * n_occupied * (n_occupied+1)/2 * sizeof(double));
    memset(temp5, 0, nmo * n_occupied * n_occupied * (n_occupied+1)/2 * sizeof(double));
    double* temp6 = (double*) malloc(nmo * n_occupied * n_occupied * (n_occupied+1)/2 * sizeof(double));
    memset(temp6, 0, nmo * n_occupied * n_occupied * (n_occupied+1)/2 * sizeof(double));   
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
        int stride2 = kl_up * 2;
        int k = index_map_kl[stride2];
        int l = index_map_kl[stride2+1];
        int kl = k * n_occupied + l;
        int lk = l * n_occupied + k;
        for (int pm = 0; pm < nmo * n_occupied; pm++) {
            temp5[kl_up * nmo * n_occupied + pm] = temp3[pm * n_occupied * n_occupied + kl];
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, n_occupied, n_occupied, 1.0, temp5+kl_up*nmo*n_occupied,
                  n_occupied, U, nmo, 0.0,
                  temp6+kl_up*nmo*n_occupied, n_occupied);
    }
    cblas_dcopy( nmo * n_occupied * n_occupied * (n_occupied+1)/2,temp6,1,temp5,1);
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied, n_occupied, n_occupied, 1.0, U,
                  nmo, temp5+kl_up*nmo*n_occupied, n_occupied, 0.0,
                  temp6+kl_up*nmo*n_occupied, n_occupied);
    }
    #pragma omp parallel for num_threads(16)
    for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
        int stride2 = kl_up * 2;
        int k = index_map_kl[stride2];
        int l = index_map_kl[stride2+1];
	int kl = k * n_occupied + l;
	int lk = l * n_occupied + k;
        for (int p = 0; p < nmo; p++) {
            for (int m = 0; m < n_occupied; m++) {
		int lm = l * n_occupied + m;
		int ml = m * n_occupied + l;
		int km = k * n_occupied + m;
		int mk = m * n_occupied + k;
	        J1[kl * nmo * nmo + p * nmo + m] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	        J1[lk * nmo * nmo + p * nmo + m] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	        J1[kl * nmo * nmo + m * nmo + p] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	        J1[lk * nmo * nmo + m * nmo + p] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	        K1[lm * nmo * nmo + k * nmo + p] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	        K1[km * nmo * nmo + l * nmo + p] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	        K1[ml * nmo * nmo + p * nmo + k] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	        K1[mk * nmo * nmo + p * nmo + l] = temp6[kl_up * nmo * n_occupied + p * n_occupied + m];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride = ab_up * 2;
        int a = index_map_ab[stride];
        int b = index_map_ab[stride+1];
	//int pq = p * nmo + q;
	//printf("%4d%4d%4d\n", pq_up, p , q);
        for (int k = 0; k < n_occupied; k++) {
            for (int l = 0; l < n_occupied; l++) {
		int kl = k * n_occupied + l;
	        J_half[ab_up * n_occupied * n_occupied + kl] = K[kl * nmo * nmo + (a + n_occupied) * nmo + b + n_occupied];
	    }
	}
    }
    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride = ab_up * 2;
        int a = index_map_ab[stride];
        int b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied, n_occupied, n_occupied, 1.0, J_half+ab_up*n_occupied*n_occupied,
                  n_occupied, U, nmo, 0.0,
                  temp1+ab_up*n_occupied*n_occupied, n_occupied);
    }

    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride = ab_up * 2;
        int a = index_map_ab[stride];
        int b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied, n_occupied, n_occupied, 1.0, U,
                  nmo, temp1+ab_up*n_occupied*n_occupied, n_occupied, 0.0,
                  temp2+ab_up*n_occupied*n_occupied, n_occupied);
    }
    #pragma omp parallel for num_threads(16)
    for (int ab_up = 0; ab_up < n_virtual*(n_virtual+1)/2; ab_up++) {
	int stride1 = ab_up * 2;
        int a = index_map_ab[stride1];
        int b = index_map_ab[stride1+1];
        for (int kl_up = 0; kl_up < n_occupied*(n_occupied+1)/2; kl_up++) {
            int stride2 = kl_up * 2;
            int k = index_map_kl[stride2];
            int l = index_map_kl[stride2+1];
	    int kl = k * n_occupied + l;
	    int lk = l * n_occupied + k;
	    K1[kl * nmo * nmo + (a + n_occupied) * nmo + b + n_occupied] = temp2[ab_up * n_occupied * n_occupied + kl];
	    K1[lk * nmo * nmo + (a + n_occupied) * nmo + b + n_occupied] = temp2[ab_up * n_occupied * n_occupied + lk];
	    K1[kl * nmo * nmo + (b + n_occupied) * nmo + a + n_occupied] = temp2[ab_up * n_occupied * n_occupied + lk];
	    K1[lk * nmo * nmo + (b + n_occupied) * nmo + a + n_occupied] = temp2[ab_up * n_occupied * n_occupied + kl];
	}
    }

    //double* temp14 = (double*) malloc(n_occupied * n_occupied * nmo * nmo * sizeof(double));
    //memset(temp14, 0,n_occupied * n_occupied *  nmo * nmo * sizeof(double));
    //double* temp15 = (double*) malloc(n_occupied * n_occupied * nmo * nmo * sizeof(double));
    //memset(temp15, 0,n_occupied * n_occupied *  nmo * nmo * sizeof(double));
 

 
 
    //for (int k_p = 0; k_p < n_occupied; k_p++) {
    //    for (int l_p = 0; l_p < n_occupied; l_p++) {
    //        for (int r_p = 0; r_p < nmo; r_p++) {
    //            for (int s_p = 0; s_p < nmo; s_p++) {
    //                for (int k = 0; k < n_occupied; k++) {
    //                    for (int l = 0; l < n_occupied; l++) {
    //                        for (int r = 0; r < nmo; r++) {
    //                            for (int s = 0; s < nmo; s++) {
    //    			    temp14[k_p * n_occupied * nmo * nmo +l_p *nmo * nmo + r_p * nmo + s_p] +=
    //    			    U[k * nmo + k_p] * U[l * nmo + l_p] * U[r * nmo + r_p] * U[s * nmo + s_p] * J[k * n_occupied * nmo * nmo + l * nmo * nmo + r * nmo +s];	   
    //    			    temp15[k_p * n_occupied * nmo * nmo +l_p *nmo * nmo + r_p * nmo + s_p] +=
    //    			    U[k * nmo + k_p] * U[l * nmo + l_p] * U[r * nmo + r_p] * U[s * nmo + s_p] * K[k * n_occupied * nmo * nmo + l * nmo * nmo + r * nmo +s];	   
    //    			}
    //    		    }
    //    		}
    //    	    }
    //    	}
    //        }
    //    }
    //}
    ////for (int k = 0; k < n_occupied; k++) {
    ////    for (int l = 0; l < n_occupied; l++) {
    ////        int kl = k * n_occupied + l;
    ////        for (int r = 0; r < nmo; r++) {
    ////            for (int s = 0; s < nmo; s++) {
    ////                int rs = r * nmo + s;
    ////                //printf("%20.12lf\n", temp14[kl * nmo * nmo + rs] -  J1[kl * nmo * nmo + rs]);
    ////                printf("%20.12lf %20.12lf%20.12lf\n", temp14[kl * nmo * nmo + rs],  J1[kl * nmo * nmo + rs], temp14[kl * nmo * nmo + rs]-  J1[kl * nmo * nmo + rs]);
    ////    	}
    ////        }
    ////    }
    ////}
    //for (int k = 0; k < n_occupied; k++) {
    //    for (int l = 0; l < n_occupied; l++) {
    //        int kl = k * n_occupied + l;
    //        for (int r = 0; r < nmo; r++) {
    //            for (int s = 0; s < nmo; s++) {
    //                int rs = r * nmo + s;
    //                //printf("%20.12lf\n", temp14[kl * nmo * nmo + rs] -  J1[kl * nmo * nmo + rs]);
    //                printf("%20.12lf %20.12lf%20.12lf\n", temp15[kl * nmo * nmo + rs],  K1[kl * nmo * nmo + rs], temp15[kl * nmo * nmo + rs]-  K1[kl * nmo * nmo + rs]);
    //    	}
    //        }
    //    }
    //}
    //for (int k = 0; k < n_occupied; k++) {
    //    for (int l = 0; l < n_occupied; l++) {
    //        int kl = k * n_occupied + l;
    //        for (int a = 0; a < n_virtual; a++) {
    //            for (int b = 0; b < n_virtual; b++) {
    //                int ab = (a +n_occupied)* nmo + b+n_occupied;
    //                //printf("%20.12lf\n", temp14[kl * nmo * nmo + rs] -  J1[kl * nmo * nmo + rs]);
    //                printf("%20.12lf %20.12lf%20.12lf\n", temp14[kl * nmo * nmo + ab],  J1[kl * nmo * nmo + ab], temp14[kl * nmo * nmo + ab]-  J1[kl * nmo * nmo + ab]);
    //    	}
    //        }
    //    }
    //}
    double* temp7 = (double*) malloc(nmo * nmo * sizeof(double));
    memset(temp7, 0, nmo * nmo * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, h,
                  nmo, U, nmo, 0.0,
                  temp7, nmo);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, U,
                  nmo, temp7, nmo, 0.0,
                  h1, nmo);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, d_cmo,
                  nmo, U, nmo, 0.0,
                  temp7, nmo);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo, nmo, nmo, 1.0, U,
                  nmo, temp7, nmo, 0.0,
                  d_cmo1, nmo);
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




void build_sigma_reduced(double* U, double* A_tilde, int* index_map, double* G, double* R_reduced, double* sigma_reduced, int num_states, int pointer, int nmo, int index_map_size, int n_occupied){
    double* R_total = (double*) malloc(num_states * nmo * n_occupied * sizeof(double));
    memset(R_total, 0, num_states * nmo * n_occupied * sizeof(double));
    double* sigma_total = (double*) malloc(num_states * nmo * n_occupied * sizeof(double));
    memset(sigma_total, 0, num_states * nmo * n_occupied * sizeof(double));
    //printf("num state %d\n",num_states); 
    //printf("index_map_size %d\n",index_map_size); 
    double* A3 = (double*) malloc(nmo * nmo * sizeof(double));
    memset(A3, 0, nmo * nmo * sizeof(double));
    #pragma omp parallel for num_threads(16)
    for (int p = 0; p < nmo; p++) {
        for (int q = 0; q < nmo; q++) {
	    A3[p * nmo + q] = A_tilde[p * nmo + q] + A_tilde[q * nmo + p];
	}
    }
    
    #pragma omp parallel for num_threads(16)
    for (int j = 0; j < index_map_size; j++) {
        int r = index_map[j * 2 + 0]; 
        int k = index_map[j * 2 + 1];
        //printf("%4d  %4d %4d\n",j,r,k);	
        for (int i = 0; i < num_states; i++) {
            R_total[i * nmo * n_occupied + r * n_occupied + k] = R_reduced[i * (index_map_size + pointer) + j+pointer];
        }
    } 
     
    
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < num_states; i++) {
        double* R = (double*) malloc(nmo * n_occupied * sizeof(double));
        memset(R, 0, nmo * n_occupied * sizeof(double));
        double* sigma = (double*) malloc(nmo * n_occupied * sizeof(double));
        memset(sigma, 0, nmo * n_occupied * sizeof(double));
        double* temp1 = (double*) malloc(nmo * n_occupied * sizeof(double));
        memset(temp1, 0, nmo * n_occupied * sizeof(double));
        //double* temp2  = (double*) malloc(nmo * n_occupied * sizeof(double));
        //memset(temp2, 0, nmo * n_occupied * sizeof(double));
        //double* temp3  = (double*) malloc(nmo * n_occupied * sizeof(double));
        //memset(temp3, 0, nmo * n_occupied * sizeof(double));
        //double* temp4 = (double*) malloc(nmo * n_occupied * sizeof(double));
        //memset(temp4 , 0, nmo * n_occupied * sizeof(double));
        //double* temp5  = (double*) malloc(nmo * n_occupied * sizeof(double));
        //memset(temp5, 0, nmo * n_occupied * sizeof(double));
        //double* temp6  = (double*) malloc(nmo * n_occupied * sizeof(double));
        //memset(temp6, 0, nmo * n_occupied * sizeof(double));
        //double* temp7  = (double*) malloc(nmo * n_occupied * sizeof(double));
        //memset(temp7, 0, nmo * n_occupied * sizeof(double));
	//double* sigma2= (double*) malloc(nmo * n_occupied * sizeof(double));
        //memset(sigma2, 0, nmo * n_occupied * sizeof(double));



        cblas_dcopy(nmo * n_occupied, R_total + i*nmo*n_occupied,1,R,1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, n_occupied, nmo, 1.0, U,
                 nmo, R, n_occupied, 0.0,
                 temp1, n_occupied);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo, n_occupied, n_occupied, -1.0, U,
                 nmo, R, n_occupied, 1.0,
                 temp1, n_occupied);
        
        //for (int q = 0; q < nmo; q++) {
        //    for (int l = 0; l < n_occupied; l++) {
	//	double a = 0.0;    
        //        for (int s = 0; s < nmo; s++) {
        //            a += U[q * nmo + s] * R[s * n_occupied +l]; 
	//	}
        //            temp2[q * n_occupied + l] = a; 
	//    }
	//}	
        //for (int q = 0; q < nmo; q++) {
        //    for (int s = 0; s < n_occupied; s++) {
	//	double a = 0.0;    
        //        for (int l = 0; l < n_occupied; l++) {
        //            a += U[q * nmo + l] * R[s * n_occupied +l]; 
	//	}
        //            temp2[q * n_occupied + s] -= a; 
	//    }
	//}
        //for (int q = 0; q < nmo; q++) {
        //    for (int l = 0; l < n_occupied; l++) {
        //        printf("%20.12lf %20.12lf%20.12lf\n", temp2[q * n_occupied + l],  temp1[q * n_occupied + l], temp2[q * n_occupied + l]-  temp1[q * n_occupied + l]);
	//	    
	//    }
	//}
        //for (int p = 0; p < nmo; p++) {
        //    for (int k = 0; k < n_occupied; k++) {
	//	int pk = p * n_occupied +k;
	//	double a = 0.0;    
        //        for (int q = 0; q < nmo; q++) {
        //            for (int l = 0; l < n_occupied; l++) {
	//	        int ql = q * n_occupied +l;
        //                a += temp1[q * n_occupied + l] * G[ql * nmo * n_occupied +pk]; 
	//	    }
	//	}
        //            temp3[p * n_occupied + k] = a; 
	//    }
	//}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, nmo * n_occupied, nmo * n_occupied, 1.0, temp1,
                 nmo * n_occupied, G, nmo * n_occupied, 0.0,
                 sigma, nmo * n_occupied);
        //for (int q = 0; q < nmo; q++) {
        //    for (int l = 0; l < n_occupied; l++) {
        //        printf("%20.12lf %20.12lf%20.12lf\n", temp3[q * n_occupied + l],  sigma[q * n_occupied + l], temp3[q * n_occupied + l]-  sigma[q * n_occupied + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo, n_occupied, nmo, 1.0, U,
                 nmo, sigma, n_occupied, 0.0,
                 temp1, n_occupied);
	cblas_dcopy(nmo * n_occupied, temp1,1,sigma,1);
        //for (int r = 0; r < nmo; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
	//	double a = 0.0;    
        //        for (int p = 0; p < nmo; p++) {
        //            a += temp3[p * n_occupied + k] * U[p * nmo +r]; 
	//	}
        //            temp4[r * n_occupied + k] = a; 
	//    }
	//}
	//for (int r = 0; r < nmo; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
        //        sigma2[r * n_occupied + k] = temp4[r * n_occupied + k]; 
	//    }
	//}
        //for (int r = 0; r < n_occupied; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
        //        sigma2[r * n_occupied + k] -= temp4[k * n_occupied + r]; 
	//    }
	//}
	for (int r = 0; r < n_occupied; r++) {
            for (int k = 0; k < n_occupied; k++) {
                sigma[r * n_occupied + k] -= temp1[k * n_occupied + r]; 
	    }
	}
        //for (int q = 0; q < nmo; q++) {
        //    for (int l = 0; l < n_occupied; l++) {
        //        printf("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied + l],  sigma2[q * n_occupied + l], sigma[q * n_occupied + l]-  sigma2[q * n_occupied + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo, n_occupied, nmo, -0.5, A3,
                 nmo, R, n_occupied, 0.0,
                 temp1, n_occupied);
        cblas_daxpy(nmo * n_occupied, 1.0, temp1, 1, sigma, 1);
        //for (int r = 0; r < nmo; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
	//	double a = 0.0;    
        //        for (int s = 0; s < nmo; s++) {
        //            a += R[s * n_occupied + k] * A3[r * nmo +s]; 
	//	}
        //            temp5[r * n_occupied + k] = a; 
	//    }
	//}
        //for (int r = 0; r < nmo; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
        //        sigma2[r * n_occupied + k] -= 0.5 * temp5[r * n_occupied + k]; 
	//    }
	//}
        //for (int r = 0; r < n_occupied; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
        //        sigma2[r * n_occupied + k] += 0.5 * temp5[k * n_occupied + r]; 
	//    }
	//}
        

	for (int r = 0; r < n_occupied; r++) {
            for (int k = 0; k < n_occupied; k++) {
                sigma[r * n_occupied + k] -= temp1[k * n_occupied + r]; 
	    }
	}
        //for (int q = 0; q < nmo; q++) {
        //    for (int l = 0; l < n_occupied; l++) {
        //        printf("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied + l],  sigma2[q * n_occupied + l], sigma[q * n_occupied + l]-  sigma2[q * n_occupied + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo, n_occupied, n_occupied, 0.5, A3,
                 nmo, R, n_occupied, 1.0,
                 sigma, n_occupied);
        //for (int r = 0; r < nmo; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
	//	double a = 0.0;    
        //        for (int l = 0; l < n_occupied; l++) {
        //            a += R[k * n_occupied + l] * A3[r * nmo +l]; 
	//	}
        //            sigma2[r * n_occupied + k] += 0.5 * a; 
	//    }
	//}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo, n_occupied, n_occupied, -0.5, R,
                 n_occupied, A3, nmo, 1.0,
                 sigma, n_occupied);
        //for (int r = 0; r < nmo; r++) {
        //    for (int k = 0; k < n_occupied; k++) {
	//	double a = 0.0;    
        //        for (int l = 0; l < n_occupied; l++) {
        //            a += R[r * n_occupied + l] * A3[k * nmo +l]; 
	//	}
        //            sigma2[r * n_occupied + k] -= 0.5 * a; 
	//    }
	//}
        //for (int q = 0; q < nmo; q++) {
        //    for (int l = 0; l < n_occupied; l++) {
        //        printf("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied + l],  sigma2[q * n_occupied + l], sigma[q * n_occupied + l]-  sigma2[q * n_occupied + l]);
	//	    
	//    }
	//}
	cblas_dcopy(nmo * n_occupied, sigma,1,sigma_total + i*nmo*n_occupied,1);
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
    for (int j = 0; j < index_map_size; j++) {
        int r = index_map[j * 2 + 0]; 
        int k = index_map[j * 2 + 1]; 
        for (int i = 0; i < num_states; i++) {
            sigma_reduced[i * (index_map_size + pointer) + j+pointer] = sigma_total[i * nmo * n_occupied + r * n_occupied + k];
        }
    } 
     
free(sigma_total);
free(R_total);
free(A3);


}


