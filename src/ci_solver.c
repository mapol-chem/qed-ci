#include <stdlib.h>
#include <stdio.h>
#include "ci_solver.h"
#include <string.h>
#include <math.h>
#include <mkl.h>
#include<omp.h>
#include<time.h>
#include "mkl_lapacke.h"
#include <unistd.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BIGNUM 1E100

/* CI-solver verbosity. Numeric levels match the C++ port's casscf::PrintLevel
 * (0=Silent, 1=Normal, 2=Debug, 3=Trace, see cpp_casscf/include/casscf/logging.hpp)
 * so a caller can pass its own level straight through. The per-Davidson-iteration
 * ROOT/RESIDUAL table and iteration/timing chatter print only at Trace; at Normal
 * spin is reported by exception (only contaminated roots), and the full spin table
 * appears at Debug/Trace.
 *
 * Default is Trace: the Python path historically printed everything, and the C++
 * port's Trace level is explicitly defined to reproduce that, so defaulting high
 * keeps existing behavior until a caller opts down. Override once via the
 * QED_PRINT_LEVEL env var, or explicitly via set_ci_print_level(). */
static int ci_print_level = -1; /* -1 = not yet initialized */

int get_ci_print_level(void) {
    if (ci_print_level < 0) {
        const char* env = getenv("QED_PRINT_LEVEL");
        ci_print_level = (env != NULL) ? atoi(env) : 3; /* default Trace */
    }
    return ci_print_level;
}

void set_ci_print_level(int level) {
    ci_print_level = level;
}

/* Number of Davidson iterations the most recent CI solve took (set at the
 * converged/maxiter exit of davidson_spin/davidson). A module global + getter
 * rather than an extra constint/constdouble slot, so the Python caller -- which
 * allocates constint as exactly np.zeros(9) -- needs no change and cannot
 * overflow. Read immediately after a synchronous get_roots() call (no concurrent
 * CI solves at that level, so no race). */
static int last_ci_iterations = 0;

int get_last_ci_iterations(void) {
    return last_ci_iterations;
}

/* Per-root residual norms of the most recent CI solve. Same global+getter
 * approach as last_ci_iterations (constdouble[4] only carries the root-AVERAGED
 * residual, so per-root norms have no return channel otherwise). Capacity 64 is
 * far above the ~10 roots any run uses. */
#define CI_MAX_ROOTS 64
static double last_ci_root_residuals[CI_MAX_ROOTS] = {0.0};
static int last_ci_nroots = 0;

double get_last_ci_root_residual(int root) {
    if (root < 0 || root >= last_ci_nroots || root >= CI_MAX_ROOTS) return 0.0;
    return last_ci_root_residuals[root];
}


void matrix_product(double* A, double* B, double* C, int m, int n, int k) {
     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
}
void print_matrix(double *S, int rows, int cols, int cols_per_line) {
    if (cols_per_line <= 0 || cols_per_line > cols) {
        cols_per_line = cols;  // Default to all columns if invalid
    }
    
    int num_blocks = (cols + cols_per_line - 1) / cols_per_line;  // Ceiling division
    
    for (int block = 0; block < num_blocks; block++) {
        int start_col = block * cols_per_line;
        int end_col = (start_col + cols_per_line < cols) ? start_col + cols_per_line : cols;
        
        // Print column headers
        printf("     ");  // Space for row index
        for (int b = start_col; b < end_col; b++) {
            printf("      Col %4d    ", b);
        }
        printf("\n");
        
        // Print separator
        printf("     ");
        for (int b = start_col; b < end_col; b++) {
            printf("------------------");
        }
        printf("\n");
        
        // Print matrix rows
        for (int a = 0; a < rows; a++) {
            printf("%4d ", a);  // Row index
            for (int b = start_col; b < end_col; b++) {
                printf("%18.12lf", S[a * cols + b]);
            }
            printf("\n");
        }
        
        if (block < num_blocks - 1) {
            printf("\n");  // Blank line between blocks
        }
    }
}
int binomialCoeff(int n, int k)
{
    int C[k + 1];
    memset(C, 0, sizeof(C));
 
    C[0] = 1; // nC0 is 1
 
    for (int i = 1; i <= n; i++) {
        // Compute next row of pascal triangle using
        // the previous row
        for (int j = MIN(i, k); j > 0; j--)
            C[j] = C[j] + C[j - 1];
    }
    return C[k];
}


void get_graph(size_t N, size_t n_o, int* Y) {
    
    //lexical ordering graph with unoccupied arc set to be zero, return vertex weight
    size_t rows = (N+1)*(n_o-N+1);
    size_t cols = 3;
    size_t rows1 = N+1;
    size_t cols1 = n_o+1;
 

    int* graph = (int*) malloc(rows*cols*sizeof(int));
    memset(graph, 0, rows*cols*sizeof(int));

    int* graph_big = (int*) malloc(rows1*cols1*sizeof(int));
    memset(graph_big, 0, rows1*cols1*sizeof(int));


    graph_big[N*cols1+n_o]=1;


    //weight of vertex
    for (int e = N; e>=0; e--) { 
        for (int o = n_o-1; o>=0; o--) {
            if (e==N && o>=e){
                graph_big[e*cols1+o] = graph_big[e*cols1+o+1];
            }
            else if (e<=N && o<e){
                graph_big[e*cols1+o] = 0;
            }
            else{
                graph_big[e*cols1+o] = graph_big[(e+1)*cols1+ o+1] + graph_big[e*cols1+o+1];
            }
        }
    } 
    
    size_t count = 0;
    for (int e = 0; e<=N; e++) {
        for (int o = 0; o<=n_o; o++) {
            if (graph_big[e*cols1+o] !=0){
                graph[count*3+0] = e;
                graph[count*3+1] = o;
                graph[count*3+2] = graph_big[e*cols1+o];
                count +=1;
            }
        }
    }
    	   
    rows = N*(n_o-N+1);
    for (int row = 0; row < rows; row++) {
        int e = graph[row*3+0];   
        int o = graph[row*3+1];
        int B[1][3];
        if (e == N) {
            continue;
        }
        int i = o - e;
        int c = 0;
        if (i == 0) {
            c = 0;
            B[0][0]=e;
            B[0][1]=o;
            B[0][2]=c;
        }
        else {
            for (int j =1; j < i+1; j++) {
                c += graph_big[(e+1)*cols1+o+2-j]; 
            }
            B[0][0]=e;
            B[0][1]=o;
            B[0][2]=c;
        }
        Y[row*3+0]=B[0][0];
        Y[row*3+1]=B[0][1];
        Y[row*3+2]=B[0][2];
    } 
    free(graph);
    free(graph_big);
}


void string_to_binary(size_t string, size_t n_o) {
    int a[n_o];
    memset(a, 0, sizeof a);
    int c=0;
    while(string > 0) {
        size_t i = string%2;
        a[c++]=i; 		 
        string = (string-i)/2;  		     
    }
   
    //for (int i = 0; i < n_o; i++) {
    //        printf("%d",a[i]);
    //    	   fflush(stdout);
    //}
    //        printf("\n");
}

int* string_to_obtlist(size_t string, int nmo, int* length) {
  
     int* obts = (int*) malloc(nmo*sizeof(int));
     memset(obts, 0, nmo*sizeof(int));
     int i = 0;
     int count = 0;
     while (string != 0) {
         if ((string & 1) == 1) {
             obts[count] = i;
             count += 1;
	 }
         string >>= 1;
         i += 1;
     }
     *length=count;
     obts = (int*) realloc(obts,count*sizeof(int));

     return obts;
}


int string_to_index(size_t string, size_t N, size_t n_o, int* Y) {
     int a[n_o];
     memset(a, 0, sizeof a);
     int c=0;
     //while(string > 0) {
     //    size_t i = string%2;
     //    a[c++]=i; 		 
     //    string = (string-i)/2;  		     
     //}
     //int b[n_o];
     //memset(b, 0, sizeof b);
     //c=0;
     while(string > 0) {
         a[c++]= string & 1; 		 
         string >>= 1UL;  		     
     }
     
     
     int count = 0;
     int index = 0;
     int rows = N*(n_o-N+1);
     

     for (int i = 0; i < n_o; i++) {
         if (a[i] == 1) {
             int e = count;
             int o = i;
             for (int j = 0; j < rows; j++) {
                 if  (Y[j*3+0] == e && Y[j*3+1] == o) {
                     index +=Y[j*3+2];
		 }
	     }
             count +=1;
	 }	
     } 
      
     return index;
} 

size_t index_to_string(int index, int N, int n_o, int* Y) {
    int index_sum=0;
    int e=N;
    int o=n_o;
    int rows = N*(n_o-N+1);
    int count=0;
    int arr[n_o];
    memset(arr, 1, sizeof arr);
    int count3=0;
    while(index_sum<=index && e <= o) {
        if (e == o && index_sum<index) {
            int count2 = 0;
            for (int i = 0; i < count3; i++) {
                if (arr[n_o-count3+i] == 1) {
                   arr[n_o-count3+i] = 0;
                   count2 = i;
                   o = o+i;
                   e = e+1;
                   break;
     	   }
            }
            for (int j = 0; j < rows; j++) {
                if  (Y[j*3+0] == e-1 && Y[j*3+1] == o) {
                    int b = Y[j*3+2];
                    index_sum = index_sum-b;
     	   }
            }
            count3 -= count2;
        }

        else {
            if (e > 0) {
                for (int j = 0; j < rows; j++) {
                    if (Y[j*3+0] == e-1 && Y[j*3+1] == o-1) {
                        int a = Y[j*3+2];
                        if (a <= index-index_sum) {
                            e = e-1;
                            o = o-1;
                            index_sum = index_sum+a;
                            arr[n_o-count3-1] = 1;
                            count3 += 1;
     		   }
                        else {
                            o = o-1;
                            arr[n_o-count3-1] = 0;
                            count3 += 1;
     		   }
     	       }
     	   }
            }
            else if (e == 0) {
                o = o-1;
                arr[n_o-count3-1] = 0;
                count3 += 1;
            }
        }
        count += 1;
        if (count == 1500000 || (e==0 && o==0 && index_sum==index))  break;
    }
    size_t string = 0;
    for (int i = 0; i < n_o; i++) {
        if (arr[i] == 1) {
            //string += pow(2,i);
	    string |= (size_t)1 << i;

        }
    }


return string;
   
}
int phase_single_excitation(size_t p, size_t q, size_t string) {
       size_t mask;
       if (p>q) {
           mask=(1UL<<p)-(1UL<<(q+1));
       }
       else {
           mask=(1UL<<q)-(1UL<<(p+1));
       }
       if (__builtin_popcountll(mask & string) %2) {
           return -1;
       }
       else {
           return 1;
       }
}

void single_creation_list2(int N_ac, int n_o_ac, int n_o_in, int* Y,int* table_creation) {
    int num_strings = binomialCoeff(n_o_ac, N_ac-1);
    int rows = (N_ac-1)*(n_o_ac-N_ac+2);
    int cols = 3;
         

    int* Y1 = (int*) malloc(rows*cols*sizeof(int));

    get_graph(N_ac-1, n_o_ac, Y1); 
         
    int count=0;        
    for (int index = 0; index < num_strings; index++){
        size_t string = index_to_string(index,N_ac-1,n_o_ac,Y1);
	//string_to_binary(string, n_o_ac);
	//printf("\n");
        int vir[n_o_ac-N_ac+1];
        int sign[n_o_ac-N_ac+1];
        int count_vir = 0;
        for (int i = 0; i < n_o_ac; i++) {
            if ((string &(1UL<<i)) == 0) {
                vir[count_vir] = i;
                size_t mask = (1UL<<i)-1;
		if (((__builtin_popcountll(mask & string)) + n_o_in)  %2) {
		    sign[count_vir] = -1;
		}
		else {
		    sign[count_vir] = 1;
		}
     	        count_vir++;
            }
        }
        
        for (int a = 0; a < n_o_ac-N_ac+1; a++) {
            size_t string1 = string | (1UL<<vir[a]);
            table_creation[count*3+0] = string_to_index(string1,N_ac,n_o_ac,Y);
            table_creation[count*3+1] = sign[a];
            table_creation[count*3+2] = vir[a];
            count += 1;
        }
    }
    free(Y1);
}

void single_annihilation_list2(int N_ac, int n_o_ac, int n_o_in, int* Y, int* table_annihilation) {
    int num_strings = binomialCoeff(n_o_ac, N_ac);
    int rows = (N_ac-1)*(n_o_ac-N_ac+2);
    int cols = 3;
         
    //printf("%4d\n",num_strings);
    int* Y1 = (int*) malloc(rows*cols*sizeof(int));

    get_graph(N_ac-1, n_o_ac, Y1); 
         
    int count=0;        
    for (int index = 0; index < num_strings; index++){
        size_t string = index_to_string(index,N_ac,n_o_ac,Y);
	//string_to_binary(string, n_o_ac);
	//printf("\n");
        int occ[N_ac];
        int count_occ = 0;
        for (int i = 0; i < n_o_ac; i++) {
            if (string &(1UL<<i)) {
                occ[count_occ] = i;
     	        count_occ++;
            }
        }
	//printf("\n");
        
        for (int i = 0; i < N_ac; i++) {
            size_t string1 = string ^ (1UL<<occ[i]);
	    int sign;
	    if ((i+n_o_in)%2) {
	        sign = -1;
	    }
	    else {
		sign = 1;    
	    }
            table_annihilation[count*3+0] = string_to_index(string1,N_ac-1,n_o_ac,Y1);
            table_annihilation[count*3+1] = sign;
            table_annihilation[count*3+2] = occ[i];
            count += 1;
        }
    }
    free(Y1);
}


void single_replacement_list(int num_alpha, int N_ac, int n_o_ac, int* Y,int* table) {

       int count=0;        
       for (int index = 0; index < num_alpha; index++){
           size_t string = index_to_string(index,N_ac,n_o_ac,Y);
	   //string_to_binary(string, n_o_ac);
           int occ[N_ac];
           int vir[n_o_ac-N_ac];
           int count_occ = 0;
           int count_vir = 0;
           for (int i = 0; i < n_o_ac; i++) {
               if (string &(1UL<<i)) {
                   occ[count_occ] = i;
	           count_occ++;
	       }
               else {
                   vir[count_vir] = i;
		   count_vir++;
	       }
	   }
           
           for (int i = 0; i < N_ac; i++) {
               table[count*4+0] = index;
               table[count*4+1] = 1; 
               table[count*4+2] = occ[i]; 
               table[count*4+3] = occ[i];
               count += 1;
	   }
           for (int i = 0; i < N_ac; i++) {
               for (int a = 0; a < n_o_ac-N_ac; a++) {
                   uint64_t string1 = (string^(1UL<<occ[i])) | (1UL<<vir[a]);
                   table[count*4+0] = string_to_index(string1,N_ac,n_o_ac,Y);
                   table[count*4+1] = phase_single_excitation(vir[a],occ[i],string);
                   table[count*4+2] = vir[a];
                   table[count*4+3] = occ[i];
                   count += 1;
	       }
	   }
       }
}


void build_H_diag(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha,int nmo, int n_act_a,int n_act_orb,int n_in_a, double omega, double Enuc, double dc, int* Y) {
    size_t num_dets = num_alpha * num_alpha;
    int np1 = N_p + 1;
    #pragma omp parallel for
    for (size_t index_photon_det = 0; index_photon_det < np1*num_dets; index_photon_det++) {
        size_t Idet = index_photon_det%num_dets;	
        int m = (index_photon_det-Idet)/num_dets;	
        int start =  m * num_dets; 
    
        int index_b = Idet%num_alpha;
        int index_a = (Idet-index_b)/num_alpha;
        size_t string_a = index_to_string(index_a,n_act_a,n_act_orb,Y);
        size_t string_b = index_to_string(index_b,n_act_a,n_act_orb,Y);
    	size_t ab = string_a & string_b;
    	int length;
    	int* double_active = string_to_obtlist(ab, nmo, &length);
    	int dim_d = length+n_in_a;
    	int double_occupation[dim_d];
        for (int i = 0; i < dim_d; i++) {
            if (i < n_in_a) {
    	       double_occupation[i] = i;	    
    	    }
            else {
    	       double_occupation[i] = double_active[i-n_in_a] + n_in_a;	
    	    }
    	}
            
        size_t e = string_a^string_b;
        size_t ea = e&string_a;
        size_t eb = e&string_b;
        int dim_s;
    	int* single_occupation_a = string_to_obtlist(ea, nmo, &dim_s);
    	int* single_occupation_b = string_to_obtlist(eb, nmo, &dim_s);
            for (int i = 0; i < dim_s; i++) {
    	    single_occupation_a[i] += n_in_a;	
    	    single_occupation_b[i] += n_in_a;	
    	}
            
            int* occupation_list_spin = (int*) malloc((dim_d+2*dim_s)*3*sizeof(int));
            memset(occupation_list_spin, 0, (dim_d+2*dim_s)*3*sizeof(int));
            for (int i = 0; i < dim_d; i++) {
                occupation_list_spin[i*3+0] = double_occupation[i];
                occupation_list_spin[i*3+1] = 1; 
                occupation_list_spin[i*3+2] = 1; 
    	}
            for (int i = 0; i < dim_s; i++) {
                occupation_list_spin[(i+dim_d)*3+0] = single_occupation_a[i];
                occupation_list_spin[(i+dim_d)*3+1] = 1; 
                occupation_list_spin[(i+dim_d+dim_s)*3+0] = single_occupation_b[i];
                occupation_list_spin[(i+dim_d+dim_s)*3+2] = 1; 
    	}
    	
	double c = 0;
        for (int a = 0; a < (dim_d+2*dim_s); a++) {
            int i = occupation_list_spin[a*3+0];
            int n_ia = occupation_list_spin[a*3+1]; 
            int n_ib = occupation_list_spin[a*3+2];
            int n_i = n_ia+ n_ib; 
            int ii = i * nmo + i;
            c += n_i * h1e[i*nmo+i];
            for (int b = 0; b < (dim_d+2*dim_s); b++) {
                int j = occupation_list_spin[b*3+0];
                int n_ja = occupation_list_spin[b*3+1]; 
                int n_jb = occupation_list_spin[b*3+2];
                int n_j = n_ja + n_jb; 
                int jj = j * nmo + j;
                c += 0.5 * n_i * n_j * h2e[ii*nmo*nmo+jj];
                int ij = i * nmo + j;
                c -= 0.5 * (n_ia * n_ja + n_ib * n_jb) * h2e[ij*nmo*nmo+ij];
    	    }
    	}
    	//for (int i = 0; i < dim_d + 2*dim_s; i++) {
    	//    for (int j = 0; j < 3; j++) {
    	//        printf("%4d", occupation_list_spin[i*3+j]);
    	//    }
    	//        printf("\n");
     	//    
    	//}
    	//printf("\n");
        H_diag[Idet+start] = c + m * omega + Enuc + dc;
    	free(double_active);
    	free(single_occupation_a);
    	free(single_occupation_b);
    	free(occupation_list_spin);
    }

}
void build_H_diag_cas(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha,int nmo, int n_act_a,int n_act_orb,int n_in_a, double E_core, double omega, double Enuc, double dc, int* Y) {
    size_t num_dets = num_alpha * num_alpha;
    int np1 = N_p + 1;
    int n_occupied = n_act_orb + n_in_a;
    #pragma omp parallel for
    for (size_t index_photon_det = 0; index_photon_det < np1*num_dets; index_photon_det++) {
        size_t Idet = index_photon_det%num_dets;	
        int m = (index_photon_det-Idet)/num_dets;	
        int start =  m * num_dets; 
    
        int index_b = Idet%num_alpha;
        int index_a = (Idet-index_b)/num_alpha;
        size_t string_a = index_to_string(index_a,n_act_a,n_act_orb,Y);
        size_t string_b = index_to_string(index_b,n_act_a,n_act_orb,Y);
    	size_t ab = string_a & string_b;
    	int length;
    	int* double_active = string_to_obtlist(ab, nmo, &length);
    	int dim_d = length;
    	int double_occupation[dim_d];
        for (int i = 0; i < dim_d; i++) {
            //if (i < n_in_a) {
    	    //   double_occupation[i] = i;	    
    	    //}
            //else {
    	    //   double_occupation[i] = double_active[i-n_in_a] + n_in_a;	
    	    //}
    	       double_occupation[i] = double_active[i] + n_in_a;	
    	}
        
        size_t e = string_a^string_b;
        size_t ea = e&string_a;
        size_t eb = e&string_b;
        int dim_s;
    	int* single_occupation_a = string_to_obtlist(ea, nmo, &dim_s);
    	int* single_occupation_b = string_to_obtlist(eb, nmo, &dim_s);
            for (int i = 0; i < dim_s; i++) {
    	    single_occupation_a[i] += n_in_a;	
    	    single_occupation_b[i] += n_in_a;	
    	}
            
            int* occupation_list_spin = (int*) malloc((dim_d+2*dim_s)*3*sizeof(int));
            memset(occupation_list_spin, 0, (dim_d+2*dim_s)*3*sizeof(int));
            for (int i = 0; i < dim_d; i++) {
                occupation_list_spin[i*3+0] = double_occupation[i];
                occupation_list_spin[i*3+1] = 1; 
                occupation_list_spin[i*3+2] = 1; 
    	}
            for (int i = 0; i < dim_s; i++) {
                occupation_list_spin[(i+dim_d)*3+0] = single_occupation_a[i];
                occupation_list_spin[(i+dim_d)*3+1] = 1; 
                occupation_list_spin[(i+dim_d+dim_s)*3+0] = single_occupation_b[i];
                occupation_list_spin[(i+dim_d+dim_s)*3+2] = 1; 
    	}
    	//for (int i = 0; i < dim_d + 2*dim_s; i++) {
    	//    for (int j = 0; j < 3; j++) {
    	//        printf("%4d", occupation_list_spin[i*3+j]);
    	//    }
    	//        printf("\n");
     	//    
    	//}
    	//        printf("\n");
        double c = 0;
        for (int a = 0; a < (dim_d+2*dim_s); a++) {
            int i = occupation_list_spin[a*3+0];
            int n_ia = occupation_list_spin[a*3+1]; 
            int n_ib = occupation_list_spin[a*3+2];
            int n_i = n_ia+ n_ib; 
            int ii = i * n_occupied + i;
            c += n_i * h1e[i*n_occupied+i];
            for (int b = 0; b < (dim_d+2*dim_s); b++) {
                int j = occupation_list_spin[b*3+0];
                int n_ja = occupation_list_spin[b*3+1]; 
                int n_jb = occupation_list_spin[b*3+2];
                int n_j = n_ja + n_jb; 
                int jj = j * n_occupied + j;
                c += 0.5 * n_i * n_j * h2e[ii*n_occupied*n_occupied+jj];
                int ij = i * n_occupied + j;
                c -= 0.5 * (n_ia * n_ja + n_ib * n_jb) * h2e[ij*n_occupied*n_occupied+ij];
    	    }
    	}
	//printf("%20.12lf %20.12lf  %20.12lf \n", pp, qq, pp + qq + E_core);
        H_diag[Idet+start] = c + m * omega + Enuc + dc + E_core;
    	free(double_active);
    	free(single_occupation_a);
    	free(single_occupation_b);
    	free(occupation_list_spin);
    }

}
void build_H_diag_cas_spin(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha,int nmo, int n_act_a,int n_act_orb,int n_in_a, double E_core, double omega, double Enuc, 
		double dc, int* Y, double target_spin) {
    size_t num_dets = num_alpha * num_alpha;
    int np1 = N_p + 1;
    int n_occupied = n_act_orb + n_in_a;
    /* QED_HDIAG_STRTAB (2026-08-20, item 241).  index_to_string() depends only
       on the alpha (or beta) INDEX, of which there are num_alpha = 252 at c10h8
       dims -- but the loop below called it twice for every one of the
       num_alpha^2 = 63504 determinants, i.e. 127008 calls for 252 distinct
       answers.  Tabulate it once.  Bit-identical: same function, same
       arguments, same results. */
    size_t* str_tab = (size_t*) malloc((size_t) num_alpha * sizeof(size_t));
    for (int idx = 0; idx < num_alpha; idx++) {
        str_tab[idx] = index_to_string(idx, n_act_a, n_act_orb, Y);
    }
    #pragma omp parallel for
    for (size_t index_photon_det = 0; index_photon_det < np1*num_dets; index_photon_det++) {
    	//printf("%d\n", index_photon_det);
        size_t Idet = index_photon_det%num_dets;	
        int m = (index_photon_det-Idet)/num_dets;	
        int start =  m * num_dets; 
    
        int index_b = Idet%num_alpha;
        int index_a = (Idet-index_b)/num_alpha;
        size_t string_a = str_tab[index_a];   /* QED_HDIAG_STRTAB */
        size_t string_b = str_tab[index_b];
    	size_t ab = string_a & string_b;
    	int length;
    	int* double_active = string_to_obtlist(ab, nmo, &length);
    	int dim_d = length;
    	int double_occupation[dim_d];
        for (int i = 0; i < dim_d; i++) {
            //if (i < n_in_a) {
    	    //   double_occupation[i] = i;	    
    	    //}
            //else {
    	    //   double_occupation[i] = double_active[i-n_in_a] + n_in_a;	
    	    //}
    	       double_occupation[i] = double_active[i] + n_in_a;	
    	}
        
        size_t e = string_a^string_b;
        size_t ea = e&string_a;
        size_t eb = e&string_b;
        int dim_s;
    	int* single_occupation_a = string_to_obtlist(ea, nmo, &dim_s);
    	int* single_occupation_b = string_to_obtlist(eb, nmo, &dim_s);
            for (int i = 0; i < dim_s; i++) {
    	    single_occupation_a[i] += n_in_a;	
    	    single_occupation_b[i] += n_in_a;	
    	}
            
            int* occupation_list_spin = (int*) malloc((dim_d+2*dim_s)*3*sizeof(int));
            memset(occupation_list_spin, 0, (dim_d+2*dim_s)*3*sizeof(int));
            for (int i = 0; i < dim_d; i++) {
                occupation_list_spin[i*3+0] = double_occupation[i];
                occupation_list_spin[i*3+1] = 1; 
                occupation_list_spin[i*3+2] = 1; 
    	}
            for (int i = 0; i < dim_s; i++) {
                occupation_list_spin[(i+dim_d)*3+0] = single_occupation_a[i];
                occupation_list_spin[(i+dim_d)*3+1] = 1; 
                occupation_list_spin[(i+dim_d+dim_s)*3+0] = single_occupation_b[i];
                occupation_list_spin[(i+dim_d+dim_s)*3+2] = 1; 
    	}
    	//for (int i = 0; i < dim_d + 2*dim_s; i++) {
    	//    for (int j = 0; j < 3; j++) {
    	//        printf("%4d", occupation_list_spin[i*3+j]);
    	//    }
    	//        printf("\n");
     	//    
    	//}
        //fflush(stdout);
    	//printf("\n");
    	//printf("\n");
    	//printf("\n");
 	double F = 0.0;
        if (dim_s != 0) {	
	    F = (4.0*target_spin * target_spin -2.0*dim_s) / (2.0*dim_s) / (2.0*dim_s-1.0);
        }
	//printf("%20.12lf\n", F);
        double c = 0.0;
        for (int a = 0; a < (dim_d+2*dim_s); a++) {
            int i = occupation_list_spin[a*3+0];
            int n_ia = occupation_list_spin[a*3+1]; 
            int n_ib = occupation_list_spin[a*3+2];
            int n_i = n_ia+ n_ib;
	    int N_i = n_i * (2 - n_i);
            int ii = i * n_occupied + i;
            c += n_i * h1e[ii];
	    if (target_spin >= 0.0) {
	        c -= 0.25 * N_i * h2e[ii*n_occupied*n_occupied+ii];
	    }
            for (int b = 0; b < (dim_d+2*dim_s); b++) {
                int j = occupation_list_spin[b*3+0];
                int n_ja = occupation_list_spin[b*3+1]; 
                int n_jb = occupation_list_spin[b*3+2];
                int n_j = n_ja + n_jb; 
		int N_j = n_j * (2 - n_j);
		int jj = j * n_occupied + j;
		int ij = i * n_occupied + j;
		//printf("%4d %4d %4d %4d \n", n_i, N_i, n_j, N_j);
                c += 0.5 * n_i * n_j * h2e[ii*n_occupied*n_occupied+jj];
	        if (target_spin < 0.0) {
                    c -= 0.5 * (n_ia * n_ja + n_ib * n_jb) * h2e[ij*n_occupied*n_occupied+ij];
	        }
	        if (target_spin >= 0.0) {
                    c -= 0.25 * n_i * n_j * h2e[ij*n_occupied*n_occupied+ij];
                    if (i !=j) {
	                c -= 0.25 * F * N_i * N_j * h2e[ij*n_occupied*n_occupied+ij];
	            }
	        }
    	    }
    	}
	//printf("%20.12lf %20.12lf %20.12lf %20.12lf\n",pp,qq,c, pp + qq + E_core);
        H_diag[Idet+start] = c + m * omega + Enuc + dc + E_core;
    	//printf("%4d%20.12lf\n",2*dim_s, H_diag[Idet+start]);

	free(double_active);
    	free(single_occupation_a);
    	free(single_occupation_b);
    	free(occupation_list_spin);
    }
    free(str_tab);                               /* QED_HDIAG_STRTAB */
    
    fflush(stdout);
}
void get_string(double* h1e, double* h2e, double* H_diag, int* b_array, int* table, int* table_creation, int* table_annihilation, int N_p, int num_alpha, int nmo, int N, 
		int n_o, int n_in_a, double E_core, double omega,double Enuc, double dc, double target_spin) {
    int rows = N*(n_o-N+1);
    int cols = 3;
    
    int* Y = (int*) malloc(rows*cols*sizeof(int));
    clock_t t;
    t = clock();
    get_graph(N,n_o,Y);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    if (get_ci_print_level() >= 3) printf("get_graph took %f seconds to execute \n", time_taken);

    rows = num_alpha*(N*(n_o-N)+N+n_in_a);
    cols = 4;
    double itime, ftime, exec_time;
    itime = omp_get_wtime();
    //build_H_diag(h1e, h2e, H_diag, N_p, num_alpha, nmo, N, n_o, n_in_a, omega, Enuc, dc,Y);   
    build_H_diag_cas_spin(h1e, h2e, H_diag, N_p, num_alpha, nmo, N, n_o, n_in_a, E_core, omega, Enuc, dc,Y, target_spin);   
    ftime = omp_get_wtime();
    time_taken = ((double)t)/CLOCKS_PER_SEC;
    exec_time = ftime - itime;
    if (get_ci_print_level() >= 3) printf("buil_H_diag took %f seconds to execute \n", exec_time);

    t = clock();
    single_replacement_list(num_alpha, N, n_o, Y, table);   
    single_creation_list2(N,n_o, n_in_a, Y,table_creation);
    single_annihilation_list2(N,n_o, n_in_a, Y,table_annihilation);
    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC;
    if (get_ci_print_level() >= 3) printf("single_replacement_list took %f seconds to execute \n", time_taken);

    int num_links = N * (n_o-N) + N;
    build_b_array(table, b_array, num_alpha, num_links, n_o);

    free(Y);
}



void build_sigma(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors,
    int* table, int* table_creation, int* table_annihilation, int N_ac, int n_o_ac, int n_o_in, int nmo,
    int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, double E_core, bool break_degeneracy) {

    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    int np1 = N_p + 1;
    // Adaptive parallelization: num_state is the CURRENT Davidson subspace
    // size (grows toward davidson_maxdim -- can be large for demanding
    // calculations with many roots/a generous subspace), so num_state*np1
    // is this loop's own trip count. Each (n,m) task is fully independent
    // (a different trial vector/photon block, no data dependency between
    // tasks) -- when that count is enough to keep every thread busy,
    // parallelize HERE exactly as before, with sigma3/sigma12/sigma_dipole
    // running serially inside (no BLAS-nesting risk at all). Otherwise
    // (few roots, small subspace -- the common case when the active space
    // itself is large and N_p is small), parallelize INSIDE those three
    // functions over num_alpha instead, which scales with active-space
    // size rather than subspace size.
    int outer_units = num_state * np1;
    int use_outer_parallel = (outer_units >= omp_get_max_threads());
    int parallel_inner = !use_outer_parallel;
    #pragma omp parallel for collapse(2) if(use_outer_parallel)
    for (int n = 0; n < num_state; n++) {
        for (int m = 0; m < np1; m++) {
            sigma12(h1e, h2e, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, n_o_ac, n_o_in, m, n, np1, parallel_inner);
            sigma3(h2e, c_vectors, c1_vectors, table, table_creation, table_annihilation,
       	   N_ac, n_o_ac, n_o_in, nmo, m, n, np1, parallel_inner);
            double someconstant = m * omega + Enuc + dc + E_core;
            if (break_degeneracy == true) {
               someconstant = m * (omega + 1) + Enuc + dc + E_core;
            }
            constant_terms_contraction(c_vectors, c1_vectors, num_alpha, someconstant, m, m, n, np1);
	    if (N_p == 0) continue;
            if ((0 < m) && (m < N_p)) {
                someconstant = -sqrt(m * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m-1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);
                someconstant = -sqrt((m+1) * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m+1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);
            }
            else if (m == N_p) {
                someconstant = -sqrt(m * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m-1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);
            }
            else {
                someconstant = -sqrt((m+1) * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m+1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);
            }
        }
    }
}

void build_sigma_2(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors,
    int* table, int* table_creation, int* table_annihilation, int N_ac, int n_o_ac, int n_o_in, int nmo,
    int num_state, int N_p, double Enuc, double dc, double omega1, double omega2, double d_exp, double E_core, bool break_degeneracy) {

    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    int np1 = N_p + 1;
    // See build_sigma's own comment -- identical adaptive-dispatch reasoning.
    int outer_units = num_state * np1;
    int use_outer_parallel = (outer_units >= omp_get_max_threads());
    int parallel_inner = !use_outer_parallel;
    #pragma omp parallel for collapse(2) if(use_outer_parallel)
    for (int n = 0; n < num_state; n++) {
        for (int m = 0; m < np1; m++) {
            sigma12(h1e, h2e, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, n_o_ac, n_o_in, m, n, np1, parallel_inner);
            sigma3(h2e, c_vectors, c1_vectors, table, table_creation, table_annihilation,
       	   N_ac, n_o_ac, n_o_in, nmo, m, n, np1, parallel_inner);
            double someconstant = m * omega1 + Enuc + dc + E_core;
            if (break_degeneracy == true) {
               someconstant = m * (omega1 + 1) + Enuc + dc + E_core;
            }
            constant_terms_contraction(c_vectors, c1_vectors, num_alpha, someconstant, m, m, n, np1);
	    if (N_p == 0) continue;
            if ((0 < m) && (m < N_p)) {
                someconstant = -sqrt(m * omega2/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m-1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);
                someconstant = -sqrt((m+1) * omega2/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m+1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);
            }
            else if (m == N_p) {
                someconstant = -sqrt(m * omega2/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m-1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);
            }
            else {
                someconstant = -sqrt((m+1) * omega2/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m+1, m, n, np1, parallel_inner);
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);
            }
        }
    }
}
 

void sigma3(double* h2e, double* c_vectors, double* c1_vectors, int* table,  int* table_creation, int* table_annihilation,
		 int N_ac, int n_o_ac, int n_o_in, int nmo, int photon_p, int state_p, int num_photon, int parallel_inner) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_alpha1 = binomialCoeff(n_o_ac, N_ac-1);
    size_t num_dets = num_alpha * num_alpha;

    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    int num_links1 = n_o_ac-N_ac+1;
    int num_links2 = N_ac;
    int n_occupied = n_o_ac + n_o_in;
    double* D = (double*) malloc(num_alpha1 * n_o_ac * num_alpha*sizeof(double));
    memset(D, 0, num_alpha1 * n_o_ac * num_alpha * sizeof(double));
    // Phase 1: each index_ka writes only to D[...*num_alpha1 + index_ka] --
    // index_ka is D's fastest-varying index, so different index_ka
    // iterations never touch the same element. Embarrassingly parallel.
    #pragma omp parallel for if(parallel_inner)
    for (int index_ka = 0; index_ka < num_alpha1; index_ka++) {
	int stride = index_ka * num_links1;
        for (int creation = 0; creation < num_links1; creation++) {
	    int index_ja = table_creation[(stride+creation)*3+0];
	    int sign = table_creation[(stride+creation)*3+1];
	    int j = table_creation[(stride+creation)*3+2];
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
	        int index_J = index_ja * num_alpha + index_jb;
                D[(index_jb * n_o_ac + j) * num_alpha1 + index_ka] += sign *
                     c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];


	    }
	}

    }

    double* T = (double*) malloc(num_alpha1 * n_o_ac * num_alpha*sizeof(double));
    memset(T, 0, num_alpha1 * n_o_ac * num_alpha * sizeof(double));
    // Phase 2: each index_ib owns its own contiguous block
    // T[index_ib*n_o_ac*num_alpha1 .. ], only reading the now-complete D
    // (built in phase 1) -- embarrassingly parallel across index_ib. Each
    // iteration calls cblas_dgemm, so MKL is pinned to 1 thread for the
    // duration of this region to avoid nested-threading oversubscription
    // (same reasoning as sigma12/sigma_dipole -- self-contained here too,
    // safe to call standalone regardless of caller).
    int mkl_threads_saved = 0;
    if (parallel_inner) {
        mkl_threads_saved = mkl_get_max_threads();
        mkl_set_num_threads(1);
    }
    #pragma omp parallel for if(parallel_inner)
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        int stride = index_ib * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int k = table[(stride + excitation)*4+2];
            int l = table[(stride + excitation)*4+3];
            int kl = (k+n_o_in) * n_occupied + (l+n_o_in);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_o_ac, num_alpha1, n_o_ac, sign, h2e+kl*n_occupied*n_occupied+n_o_in*n_occupied+n_o_in,
               	  n_occupied, D+index_jb*n_o_ac*num_alpha1, num_alpha1, 1.0,
            	  T+index_ib*n_o_ac*num_alpha1, num_alpha1);

            //for (int index_ka = 0; index_ka < num_alpha1; index_ka++) {
            //    for (int i = 0; i < n_o_ac; i++) {
            //        for (int j = 0; j < n_o_ac; j++) {
            //            int ij = (i+n_o_in) * nmo + j+n_o_in;
            //            T[(index_ib * n_o_ac + i) * num_alpha1 + index_ka] +=
	    //    		sign * h2e[kl*nmo*nmo + ij] * D[(index_jb * n_o_ac  + j) * num_alpha1 + index_ka];
	    //        }
	    //    }
	    //}

        }
    }
    if (parallel_inner) {
        mkl_set_num_threads(mkl_threads_saved);
    }


    // Phase 3: each index_ia owns its own contiguous block of c1_vectors,
    // only reading the now-complete T (built in phase 2). Embarrassingly
    // parallel across index_ia, no BLAS calls here so no MKL concern.
    //printf("\n");
    #pragma omp parallel for if(parallel_inner)
    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
        int stride = index_ia * num_links2;
        for (int annihilation = 0; annihilation < num_links2; annihilation++) {
            int index_ka = table_annihilation[(stride+annihilation)*3+0];
            int sign = table_annihilation[(stride+annihilation)*3+1];
            int i = table_annihilation[(stride+annihilation)*3+2];
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                int index_I = index_ia * num_alpha + index_ib;
                c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign *
                T[(index_ib * n_o_ac + i) * num_alpha1 + index_ka];
            }
        }

    }
//    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
//        int stride = index_ib * num_links;
//        for (int excitation = 0; excitation < num_links; excitation++) {
//            int index_jb = table1[(stride + excitation)*4+0];
//            int sign = table1[(stride + excitation)*4+1];
//            int k = table1[(stride + excitation)*4+2]; 
//            int l = table1[(stride + excitation)*4+3]; 
//            int kl = (k+n_o_in) * nmo + (l+n_o_in);
//	    double c = 0.0;
//            for (int i = 0; i < n_o_in; i++) {
//		int ii = i * nmo + i;    
//		c += h2e[ii*nmo*nmo + kl];   
//	    }
//            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
//                int index_I = index_ia * num_alpha + index_ib;
//                int index_J = index_ia * num_alpha + index_jb;
//                  c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign *
//	                    c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];
//            } 
//        }
//    }
//    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
//        int stride = index_ia * num_links;
//        for (int excitation = 0; excitation < num_links; excitation++) {
//            int index_ja = table1[(stride + excitation)*4+0];
//            int sign = table1[(stride + excitation)*4+1];
//            int k = table1[(stride + excitation)*4+2]; 
//            int l = table1[(stride + excitation)*4+3]; 
//            int kl = (k+n_o_in) * nmo + (l+n_o_in);
//	    double c = 0.0;
//            for (int i = 0; i < n_o_in; i++) {
//		int ii = i * nmo + i;    
//		c += h2e[ii*nmo*nmo + kl];   
//	    }
//            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
//                int index_I = index_ia * num_alpha + index_ib;
//                int index_J = index_ja * num_alpha + index_ib;
//                c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign * 
//    			  c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];
//            } 
//        }
//    }
//
//
//
//
//    double c = 0.0;
//    for (int i = 0; i < n_o_in; i++) {
//        for (int j = 0; j < n_o_in; j++) {
//            int ii = i * nmo +i; 
//            int jj = j * nmo +j; 
//            c += h2e[ii*nmo*nmo + jj];
//	}
//    }
//    for (size_t index_I = 0; index_I < num_dets; index_I++) {
//        c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_I];
//    }

    free(D);
    free(T);
}





void sigma12(double* h1e, double* h2e, double* c_vectors, double* c1_vectors, int num_alpha, int num_links, int* table, int nmo, int n_o_ac, int n_o_in,
		int photon_p, int state_p, int num_photon, int parallel_inner) {
    size_t num_dets = num_alpha * num_alpha;
    int n_occupied = n_o_ac + n_o_in;

    // Both loops below call cblas_dgemm once per outer iteration; pin MKL
    // to 1 thread for the duration when running with per-iteration OpenMP
    // parallelism, to avoid nested-threading oversubscription (build_sigma/
    // build_sigma_2 make the same on/off decision for the whole call, this
    // just applies it locally too so sigma12 is safe to call standalone).
    int mkl_threads_saved = 0;
    if (parallel_inner) {
        mkl_threads_saved = mkl_get_max_threads();
        mkl_set_num_threads(1);
    }

    // First loop: each index_ib computes its own local F (read-only inputs:
    // table/h1e/h2e) and writes to c1_vectors at indices {index_ib,
    // num_alpha+index_ib, 2*num_alpha+index_ib, ...} -- disjoint across
    // different index_ib values, so embarrassingly parallel. F/s_resize
    // become per-thread (allocated once per team member, not once per
    // function call) instead of one buffer reused serially across
    // iterations -- required once multiple threads touch this loop at
    // once, and numerically identical to the old shared-buffer version
    // either way (each iteration still memsets its own F to zero before
    // building it fresh).
    #pragma omp parallel if(parallel_inner)
    {
        double* F = (double*) malloc(num_alpha*sizeof(double));
        double* s_resize = (double*) malloc(num_alpha*sizeof(double));

        #pragma omp for
        for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
            memset(F, 0, num_alpha*sizeof(double));
            int stride1 = index_ib * num_links;
            for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
                int index_kb = table[(stride1 + excitation1)*4+0];
                int sign1 = table[(stride1 + excitation1)*4+1];
                int k = table[(stride1 + excitation1)*4+2];
                int l = table[(stride1 + excitation1)*4+3];
                F[index_kb] += sign1 * h1e[k*n_o_ac+l];
                int kl = (k + n_o_in) * n_occupied + (l + n_o_in);
                int stride2 = index_kb * num_links;
                for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
                    int index_jb = table[(stride2 + excitation2)*4+0];
                    int sign2 = table[(stride2 + excitation2)*4+1];
                    int i = table[(stride2 + excitation2)*4+2];
                    int j = table[(stride2 + excitation2)*4+3];
                    int ij = (i + n_o_in) * n_occupied + (j + n_o_in);
                    F[index_jb] += 0.5 * sign1 * sign2 * h2e[ij*n_occupied*n_occupied+kl];
                }
            }

            memset(s_resize, 0, num_alpha*sizeof(double));
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_alpha, 1, num_alpha, 1.0, c_vectors
       		 + (state_p * num_photon + photon_p) * num_dets, num_alpha, F, 1, 0.0, s_resize, 1);
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                int index_I = index_ia * num_alpha + index_ib;
                c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += s_resize[index_ia];
            }
        }

        free(F);
        free(s_resize);
    }


    double* c_resize = (double*) malloc(num_dets*sizeof(double));

    // Plain transpose into c_resize -- disjoint writes per index_ja, no
    // BLAS, safe to parallelize with no MKL concern.
    #pragma omp parallel for if(parallel_inner)
    for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
            int index_J = index_ja * num_alpha + index_ib;
            int index_Jt = index_ib * num_alpha + index_ja;
            c_resize[index_Jt] = c_vectors[
       	 (state_p * num_photon + photon_p) * num_dets + index_J];
        }
    }



    // Second loop: same shape as the first, over index_ia this time, each
    // iteration owning the contiguous block c1_vectors[...+index_ia*num_alpha ..].
    // Must run after the first loop (accumulates via beta=1.0 on what it
    // wrote), but is itself embarrassingly parallel across index_ia.
    #pragma omp parallel if(parallel_inner)
    {
        double* F = (double*) malloc(num_alpha*sizeof(double));

        #pragma omp for
        for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
            memset(F, 0, num_alpha*sizeof(double));
            int stride1 = index_ia * num_links;
            for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
                int index_ka = table[(stride1 + excitation1)*4+0];
                int sign1 = table[(stride1 + excitation1)*4+1];
                int k = table[(stride1 + excitation1)*4+2];
                int l = table[(stride1 + excitation1)*4+3];
                F[index_ka] += sign1 * h1e[k*n_o_ac+l];
                //print(index_kb,sign1,k,l)
                int kl = (k + n_o_in) * n_occupied + (l + n_o_in);
                int stride2 = index_ka * num_links;
                for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
                    int index_ja = table[(stride2 + excitation2)*4+0];
                    int sign2 = table[(stride2 + excitation2)*4+1];
                    int i = table[(stride2 + excitation2)*4+2];
                    int j = table[(stride2 + excitation2)*4+3];
                    int ij = (i + n_o_in) * n_occupied + (j + n_o_in);
                    F[index_ja] += 0.5 * sign1 * sign2 * h2e[ij*n_occupied*n_occupied+kl];
                }
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_alpha,1,num_alpha, 1.0, c_resize, num_alpha, F, 1, 1.0,
       		 c1_vectors+(state_p * num_photon + photon_p) * num_dets+index_ia*num_alpha, 1);
        }

        free(F);
    }

    if (parallel_inner) {
        mkl_set_num_threads(mkl_threads_saved);
    }
    free(c_resize);

}
void sigma_dipole(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int n_o_ac, int n_o_in, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon, int parallel_inner) {

     size_t num_dets = num_alpha * num_alpha;
     int n_occupied = n_o_ac + n_o_in;

     // Same shape/reasoning as sigma12 -- see that function's comments.
     int mkl_threads_saved = 0;
     if (parallel_inner) {
         mkl_threads_saved = mkl_get_max_threads();
         mkl_set_num_threads(1);
     }

     #pragma omp parallel if(parallel_inner)
     {
         double* F = (double*) malloc(num_alpha*sizeof(double));
         double* s_resize = (double*) malloc(num_alpha*sizeof(double));

         #pragma omp for
         for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
             memset(F, 0, num_alpha*sizeof(double));
             int stride1 = index_ib * num_links;
             for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
                 int index_kb = table[(stride1 + excitation1)*4+0];
                 int sign1 = table[(stride1 + excitation1)*4+1];
                 int k = table[(stride1 + excitation1)*4+2];
                 int l = table[(stride1 + excitation1)*4+3];
	         int kl = (k + n_o_in) * n_occupied + l + n_o_in;
                 F[index_kb] += sign1 * h1e[kl];
	     }


             memset(s_resize, 0, num_alpha*sizeof(double));
             cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_alpha, 1, num_alpha, someconstant, c_vectors
	        	 + (state_p * num_photon + photon_p1) * num_dets, num_alpha, F, 1, 0.0, s_resize, 1);
             for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                 int index_I = index_ia * num_alpha + index_ib;
                 c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += s_resize[index_ia];
	     }

         }

         free(F);
         free(s_resize);
     }

     double* c_resize = (double*) malloc(num_dets*sizeof(double));

     #pragma omp parallel for if(parallel_inner)
     for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
         for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
             int index_J = index_ja * num_alpha + index_ib;
             int index_Jt = index_ib * num_alpha + index_ja;
             c_resize[index_Jt] = c_vectors[
        	 (state_p * num_photon + photon_p1) * num_dets + index_J];
         }
     }

     #pragma omp parallel if(parallel_inner)
     {
         double* F = (double*) malloc(num_alpha*sizeof(double));

         #pragma omp for
         for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
             memset(F, 0, num_alpha*sizeof(double));
             int stride1 = index_ia * num_links;
             for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
                 int index_ka = table[(stride1 + excitation1)*4+0];
                 int sign1 = table[(stride1 + excitation1)*4+1];
                 int k = table[(stride1 + excitation1)*4+2];
                 int l = table[(stride1 + excitation1)*4+3];
	         int kl = (k + n_o_in) * n_occupied + l + n_o_in;
                 F[index_ka] += sign1 * h1e[kl];
	     }
             cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_alpha,1,num_alpha, someconstant, c_resize, num_alpha, F, 1, 1.0,
			 c1_vectors+(state_p * num_photon + photon_p2) * num_dets+index_ia*num_alpha, 1);

         }

         free(F);
     }

     if (parallel_inner) {
         mkl_set_num_threads(mkl_threads_saved);
     }
     free(c_resize);

}
void constant_terms_contraction(double* c_vectors,double* c1_vectors,int num_alpha, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon) {
     size_t num_dets = num_alpha * num_alpha;
     for (size_t index_I = 0; index_I < num_dets; index_I++) {
         c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += someconstant * c_vectors[(state_p * num_photon + photon_p1) * num_dets + index_I];
     }
}

void symmetric_eigenvalue_problem(double* A, int N, double* eig) {
    MKL_INT n = N, lda = N, info;	
    //double w[N];
    
    /* Solve eigenproblem.
     * dsyevd (divide & conquer) instead of dsyev (QR): same eigenpairs, 2-3.8x
     * faster measured over N = 500..4000 at both 1 and 32 threads (MKL, this
     * box).  It matters because all four live call sites scale as O(N^3) and
     * N is the Davidson subspace size -- with the small warm-start subspace
     * N is O(10) and either is free, but a large cold-start P-space guess
     * (:2307 S^2 block, :2378 H_spin block) runs at N = 200..2000.
     * QED_DSYEV=1 restores the QR path for A/B comparison.
     * NOTE: not bit-identical to dsyev, so a macroiteration count measured
     * across this change is subject to the usual FP-chaos caveat. */
    static int use_qr = -1;
    if (use_qr < 0) { const char* e = getenv("QED_DSYEV"); use_qr = (e && atoi(e) != 0) ? 1 : 0; }
    if (use_qr) info = LAPACKE_dsyev ( LAPACK_ROW_MAJOR, 'V', 'U', n, A, lda, eig);
    else        info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', n, A, lda, eig);
    /* Check for convergence */
    if( info > 0 ) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
    }
    


    //stable selection sort
    double* evrp = (double*) malloc(N*sizeof(double));
    memset(evrp, 0, N*sizeof(double));
 
    size_t i, j, jmin;

    for (j = 0; j < N-1; j++) {
        jmin = j;
        // find index of minimum eigenvalue from j to n
        for (i = j+1; i < N; i++) {
            if (eig[jmin] > eig[i] ) {
                jmin = i;
            }
        }
        //save current eigenvalue and eigenvectors associated with index jmin
        double eigr = eig[jmin];
        for (i = 0; i < N; i++) {
            evrp[i] = A[i * N + jmin];
        }
        //shift values
        while (jmin > j) {
            eig[jmin] = eig[jmin-1];
            for (i = 0; i < N; i++) {
                A[i * N + jmin] = A[i * N + jmin-1];
            }
            jmin--;
        }
        eig[j] = eigr;
        for (i = 0; i < N; i++) {
            A[i * N + j] = evrp[i];
        }
     }


    free(evrp);
}
void get_roots(double* h1e, double* h2e, double* d_cmo, double* Hdiag, double* Sdiag, double* Sdiag_projection, double* eigenvals, double* eigenvecs, int* table, int* table_creation,
	       	int* table_annihilation, int* b_array, int *constint, double *constdouble, int* index_Hdiag, bool casscf, double target_spin) {
  
    callback_ test_fn = &build_sigma;
    double itime, ftime, exec_time;
    itime = omp_get_wtime();

    if (target_spin >= 0.0) {
        davidson_spin(h1e, h2e, d_cmo, Hdiag, Sdiag, Sdiag_projection, eigenvals, eigenvecs, table, 
	    table_creation, table_annihilation, b_array, constint, constdouble, index_Hdiag, casscf, target_spin, test_fn);

    }
    else {
        davidson(h1e, h2e, d_cmo, Hdiag, eigenvals, eigenvecs, table, 
	    table_creation, table_annihilation, constint, constdouble, casscf, test_fn);
    }
    ftime = omp_get_wtime();
    exec_time = ftime - itime;
    if (get_ci_print_level() >= 3) printf("Complete Davidson in %f seconds\n", exec_time);
    fflush(stdout);

}
double check_total_spin(double* c_vector, double* Sdiag, int* table, int* b_array, int num_links0, int n_o_ac, int num_alpha, int N_p) {
    size_t H_dim = (N_p + 1) * num_alpha * num_alpha;
    double norm = cblas_ddot(H_dim, c_vector, 1, c_vector, 1);
    double* s_vector = (double*) malloc(1*H_dim*sizeof(double));
    memset(s_vector, 0, 1*H_dim*sizeof(double));
    //for (int j = 0; j < H_dim ; j++) {
    //            printf("%20.12lf\n", c_vector[j]);	    
    //}
    for (int i = 0; i < H_dim; i++) {
	s_vector[i] = c_vector[i]/sqrt(norm);    
    }
    double* newS = (double*) malloc(1*H_dim*sizeof(double));
    memset(newS, 0, 1*H_dim*sizeof(double));
    build_sigma_s_square(s_vector, newS, Sdiag, b_array, table, num_links0, n_o_ac, num_alpha, 1, N_p, 1.0);
    double total_spin = cblas_ddot(H_dim, s_vector, 1, newS, 1);
    free(newS); 
    free(s_vector); 
    return total_spin;
}

void first_order_spin_projection(double* s_vector, double* Sdiag, double* Sdiag_projection, int* table, int* b_array, int num_links0, int n_o_ac, int num_alpha,
	int N_p, double target_spin) {
        size_t H_dim = (N_p + 1) * num_alpha * num_alpha;
	double total_spin = check_total_spin(s_vector, Sdiag, table, b_array, num_links0, n_o_ac, num_alpha, N_p);
        int iteration = 0;
        double* x_vector = (double*) malloc(1*H_dim*sizeof(double));
        printf("initial residual total spin %20.12lf\n", total_spin);
        while (iteration < 20) {
            if ((fabs(fabs(total_spin) - target_spin*(target_spin+1))) < 1e-10){
                break;
	    }
	    memset(x_vector, 0, 1*H_dim*sizeof(double));
            build_sigma_s_square(s_vector, x_vector, Sdiag_projection, b_array, table, num_links0, n_o_ac, num_alpha, 1, N_p, 1.0);
            double dot_xx = cblas_ddot(H_dim, x_vector, 1, x_vector, 1);
            double dot_sx = cblas_ddot(H_dim, s_vector, 1, x_vector, 1);
	    //double norm = sqrt(dot_xx);
            double coeff = dot_sx/dot_xx;
            for (int i = 0; i < H_dim; i++) {
                s_vector[i] -= coeff * x_vector[i];
	    }
	    total_spin = check_total_spin(s_vector, Sdiag, table, b_array, num_links0, n_o_ac, num_alpha, N_p);
            iteration += 1;
            
	}
        printf("residual total spin after correction %20.12lf\n", total_spin);
	free(x_vector);
}

/* ---------------------------------------------------------------------------
 * SPIN TREATMENT SELECTION (QED_SPIN_METHOD / QED_SPIN_SCALE)
 *
 * Three MUTUALLY EXCLUSIVE ways to keep the spin-adapted Davidson on the target
 * multiplicity.  Exactly one is active; they must never be combined.
 *
 *   penalty1 (default, scale 0.45) -- first-order spin penalty.  sigma gets
 *       scale * S^2 c, via build_sigma_s_square with S_diag (built with
 *       shift = 0, helper_PFCI.py:3718-3728, so it is the plain S^2 diagonal).
 *       For a spin-pure vector this shifts the eigenvalue by scale*S(S+1).
 *
 *   penalty2 (scale 0.1) -- second-order spin penalty.  sigma gets
 *       scale * (S^2 - S(S+1))^2 c, via build_sigma_s_fourth_power with
 *       S_diag_projection (built with shift = S(S+1), helper_PFCI.py:3729-3739),
 *       which applies the shifted operator twice.  It ANNIHILATES the target
 *       spin, so it shifts nothing and only pushes contaminants up.
 *
 *   projection (no scale) -- first-order spin projection.  Nothing is added to
 *       sigma; instead first_order_spin_projection() cleans the RESIDUAL each
 *       iteration before preconditioning.
 *
 * The developer's note: penalty2 and projection sometimes work where penalty1
 * does not, which is why all three are kept selectable rather than one being
 * chosen here.
 * ------------------------------------------------------------------------- */
enum { SPIN_PENALTY1 = 1, SPIN_PENALTY2 = 2, SPIN_PROJECT = 3 };

static int qed_spin_method(void) {
    static int method = -1;
    if (method > 0) return method;
    method = SPIN_PENALTY1;
    const char* e = getenv("QED_SPIN_METHOD");
    if (e && *e) {
        if      (!strcmp(e, "penalty1")   || !strcmp(e, "1")) method = SPIN_PENALTY1;
        else if (!strcmp(e, "penalty2")   || !strcmp(e, "2")) method = SPIN_PENALTY2;
        else if (!strcmp(e, "projection") || !strcmp(e, "3")) method = SPIN_PROJECT;
        else {
            printf("\n*** QED_SPIN_METHOD='%s' is not one of penalty1 | penalty2 | projection ***\n", e);
            fflush(stdout);
            exit(1);
        }
    }
    return method;
}

/* Scale for the two penalties; ignored by the projection method.  Defaults are
 * the values the code shipped with: 0.45 for penalty1, 0.1 for penalty2. */
static double qed_spin_scale(void) {
    static double scale = -1.0;
    if (scale >= 0.0) return scale;
    scale = (qed_spin_method() == SPIN_PENALTY2) ? 0.1 : 0.45;
    const char* e = getenv("QED_SPIN_SCALE");
    if (e && *e) scale = atof(e);
    return scale;
}

/* The eigenvalue shift the active treatment puts on a SPIN-PURE vector:
 * scale*S(S+1) for penalty1; zero for penalty2 (which annihilates the target
 * spin) and for projection (which never touches sigma). */
static double qed_spin_pure_shift(double target_spin) {
    return (qed_spin_method() == SPIN_PENALTY1)
         ? qed_spin_scale() * target_spin * (target_spin + 1.0) : 0.0;
}

/* Apply the active spin treatment to a block of `nvec` sigma vectors.
 * No-op for the projection method, which acts on the residual instead. */
static void apply_spin_penalty(double* Qb, double* Sb, int nvec,
        double* Sdiag, double* Sdiag_projection, int* b_array, int* table,
        int num_links0, int n_o_ac, int num_alpha, int N_p) {
    int method = qed_spin_method();
    double scale = qed_spin_scale();
    if (method == SPIN_PENALTY1)
        build_sigma_s_square(Qb, Sb, Sdiag, b_array, table, num_links0, n_o_ac,
                             num_alpha, nvec, N_p, scale);
    else if (method == SPIN_PENALTY2)
        build_sigma_s_fourth_power(Qb, Sb, Sdiag_projection, b_array, table, num_links0,
                                   n_o_ac, num_alpha, nvec, N_p, scale);
}


/* Diagnostic spin checkpoint: run ONCE at the end of a CI Davidson solve
 * (converged, or maxiter/"runaway" exit) rather than every iteration. The
 * per-iteration check_total_spin call was ~12% of total CI time (measured on
 * MgH+ CAS(8,12), see cpp_casscf/validation/ci_phase_experiment) and the
 * convergence test never used it -- convergence is decided by residual norm
 * only. Reports each root's <S^2> once, and warns if it deviates from the
 * requested target_spin*(target_spin+1). target_spin < 0 is the "no spin
 * target" sentinel, so only the value is printed in that case. */
/* Threshold for calling a root spin-contaminated. Matches the C++ port's
 * casscf::kSpinDeviationThreshold (logging.hpp): ~5e-5 of the O(1) gap between
 * adjacent spin states, far above the ~1e-17 converged noise floor, far below
 * any physically meaningful contamination. */
#define CI_SPIN_DEVIATION_THRESHOLD 1e-4

static void report_final_spin(double* eigenvecs, int nroots, double* Sdiag, int* table,
                              int* b_array, int num_links0, int n_o_ac, int num_alpha,
                              int N_p, double target_spin) {
    int level = get_ci_print_level();
    if (level <= 0) {
        return; /* Silent: no spin diagnostics at all */
    }
    size_t H_dim = (N_p + 1) * (size_t)num_alpha * num_alpha;
    double desired = target_spin * (target_spin + 1.0);
    /* Full per-root table only at Trace (the per-CI-solve spin table is
     * Trace-shaped detail, like the per-iteration ROOT table). At Silent it is
     * off; at Normal AND Debug it reports by exception -- only a contaminated
     * root is printed -- so Debug (the level for reading the orbital-opt flow)
     * isn't buried under an "ok" table on every CI solve. */
    int verbose = (level >= 3); /* Trace: every root; Normal/Debug: exceptions only */
    int printed_header = 0;
    for (int i = 0; i < nroots; i++) {
        double total_spin = check_total_spin(eigenvecs + (size_t)i * H_dim, Sdiag, table,
                                             b_array, num_links0, n_o_ac, num_alpha, N_p);
        int deviates = (target_spin >= 0.0) &&
                       (fabs(fabs(total_spin) - desired) > CI_SPIN_DEVIATION_THRESHOLD);
        if (verbose) {
            if (!printed_header) {
                /* Field widths match the data row below (%4d %20.12lf %20.12lf   %s). */
                printf("%4s %20s %20s   %s\n", "ROOT", "WFN TOTAL SPIN <S^2>",
                       "DESIRED <S^2>", "STATUS");
                printed_header = 1;
            }
            const char* status = (target_spin < 0.0) ? "(no spin target)"
                               : deviates ? "*** WARNING: SPIN CONTAMINATION ***" : "ok";
            printf("%4d %20.12lf %20.12lf   %s\n", i, total_spin, desired, status);
        } else if (deviates) {
            /* Normal: report spin by exception only -- flag the contaminated root */
            printf("*** WARNING: CI root %d spin contamination: <S^2>=%20.12lf, "
                   "desired %20.12lf (deviation %.2e)\n",
                   i, total_spin, desired, fabs(fabs(total_spin) - desired));
        }
    }
    fflush(stdout);
}

/* ================================================================== *
 *  Direct Slater-Condon build of the CI guess blocks (Algorithm 3).   *
 *                                                                     *
 *  Replaces the L full-space sigma-vector products the guess used to  *
 *  extract H_00 (= G) and S^2_00 (= Gs): build_sigma / build_sigma_s_ *
 *  square applied to L unit vectors, then projected out via G=S*Q^T.  *
 *  Here we evaluate the L x L blocks directly from occupation-list     *
 *  differences, touching ONLY the L selected determinants.            *
 *                                                                     *
 *  Bit-faithful to build_sigma's operator + index conventions:        *
 *    1e (active core-Fock F^c):  h1e[t*n_o_ac + u]   (no inactive off) *
 *    2e chemist (tu|vw):         h2e[((t+n_in)*n_occ+(u+n_in))*n_occ^2 *
 *                                     + ((v+n_in)*n_occ+(w+n_in))]     *
 *    dipole d_tu:                d_cmo[(t+n_in)*n_occ + (u+n_in)]      *
 *    block-diagonal constant:    m*omega + Enuc + dc + E_core          *
 *    cross-block (m <-> m+-1):   -sqrt(max(m,m')*omega/2) * dipole op, *
 *                                 plus scalar -(-sqrt)*d_exp on 0-diff *
 *  Occupation lists are active-orbital indices in [0, n_o_ac).        *
 *  Determinant global index d = m*num_dets + ia*num_alpha + ib.       *
 * ================================================================== */
static inline double sc_twoe(const double* h2e, int t, int u, int v, int w, int n_occ, int n_in) {
    /* chemist (tu|vw); active orbitals offset by n_in, exactly as sigma12/sigma3 */
    size_t tu = (size_t)(t + n_in) * n_occ + (u + n_in);
    size_t vw = (size_t)(v + n_in) * n_occ + (w + n_in);
    return h2e[tu * (size_t)n_occ * n_occ + vw];
}

/* One Hamiltonian matrix element H_{bra,ket} between two determinants given in
 * decoded (bit-string + orbital-list) form.
 *
 * Extracted verbatim from build_guess_blocks_scrules' inner loop so that the
 * guess-block builder and build_sparse_init_sigma below share ONE copy of the
 * Slater-Condon rules and cannot drift apart.  The S^2 element stays with the
 * guess-block builder -- only that one needs S^2.
 *
 * Conventions preserved exactly from the original inner loop: the occupied
 * orbital lists passed in are the BRA's, while every phase is evaluated on the
 * KET's strings, and the photon-diagonal constant uses the BRA's photon block.
 * `hp` is the PHYSICAL active core-Fock, not build_sigma's modified h1e (see
 * the caller's comment where hp is built). */
static inline double sc_h_element(
        size_t SAi, size_t SBi, const int* Ai, int nAi, const int* Bi, int nBi, int mi,
        size_t SAj, size_t SBj, int mj,
        const double* hp, const double* h2e, const double* d_cmo,
        int n_o_ac, int n_occ, int n_in,
        double Enuc, double dc, double omega, double d_exp, double E_core) {
    double h = 0.0;
    int dm = mi - mj;
    size_t xa = SAi ^ SAj;
    size_t xb = SBi ^ SBj;
    int da = __builtin_popcountll(xa) / 2;
    int db = __builtin_popcountll(xb) / 2;

    if (dm == 0) {
        int diff = da + db;
        if (diff == 0) {
            double e = 0.0;
            for (int x = 0; x < nAi; x++) e += hp[Ai[x] * n_o_ac + Ai[x]];
            for (int x = 0; x < nBi; x++) e += hp[Bi[x] * n_o_ac + Bi[x]];
            for (int x = 0; x < nAi; x++)
                for (int y = x + 1; y < nAi; y++) {
                    int p = Ai[x], q = Ai[y];
                    e += sc_twoe(h2e, p, p, q, q, n_occ, n_in) - sc_twoe(h2e, p, q, q, p, n_occ, n_in);
                }
            for (int x = 0; x < nBi; x++)
                for (int y = x + 1; y < nBi; y++) {
                    int p = Bi[x], q = Bi[y];
                    e += sc_twoe(h2e, p, p, q, q, n_occ, n_in) - sc_twoe(h2e, p, q, q, p, n_occ, n_in);
                }
            for (int x = 0; x < nAi; x++)
                for (int y = 0; y < nBi; y++) {
                    int p = Ai[x], q = Bi[y];
                    e += sc_twoe(h2e, p, p, q, q, n_occ, n_in);
                }
            h = e + mi * omega + Enuc + dc + E_core;
        } else if (diff == 1) {
            int a, r, alpha;
            if (da == 1) { a = __builtin_ctzll(SAj & ~SAi); r = __builtin_ctzll(SAi & ~SAj); alpha = 1; }
            else         { a = __builtin_ctzll(SBj & ~SBi); r = __builtin_ctzll(SBi & ~SBj); alpha = 0; }
            double e = hp[r * n_o_ac + a];
            /* same-spin common orbitals give Coulomb - exchange; opposite spin Coulomb only */
            int  ns  = alpha ? nAi : nBi;
            const int* os  = alpha ? Ai : Bi;   /* same-spin occ (common, since a,r excluded by being the diff) */
            int  nos = alpha ? nBi : nAi;
            const int* oos = alpha ? Bi : Ai;   /* opposite-spin occ */
            for (int x = 0; x < ns; x++) {
                int p = os[x];
                if (p == r || p == a) continue;
                e += sc_twoe(h2e, r, a, p, p, n_occ, n_in) - sc_twoe(h2e, r, p, p, a, n_occ, n_in);
            }
            for (int x = 0; x < nos; x++) {
                int p = oos[x];
                e += sc_twoe(h2e, r, a, p, p, n_occ, n_in);
            }
            int ph = phase_single_excitation(a, r, alpha ? SAj : SBj);
            h = ph * e;
        } else if (diff == 2) {
            if (da == 2) {
                int a1 = __builtin_ctzll(SAj & ~SAi);
                int a2 = __builtin_ctzll((SAj & ~SAi) & ~(1UL << a1));
                int r1 = __builtin_ctzll(SAi & ~SAj);
                int r2 = __builtin_ctzll((SAi & ~SAj) & ~(1UL << r1));
                double val = sc_twoe(h2e, r1, a1, r2, a2, n_occ, n_in) - sc_twoe(h2e, r1, a2, r2, a1, n_occ, n_in);
                size_t s = SAj;
                int p1 = phase_single_excitation(a1, r1, s);
                s = (s & ~(1UL << a1)) | (1UL << r1);
                int p2 = phase_single_excitation(a2, r2, s);
                h = (p1 * p2) * val;
            } else if (db == 2) {
                int a1 = __builtin_ctzll(SBj & ~SBi);
                int a2 = __builtin_ctzll((SBj & ~SBi) & ~(1UL << a1));
                int r1 = __builtin_ctzll(SBi & ~SBj);
                int r2 = __builtin_ctzll((SBi & ~SBj) & ~(1UL << r1));
                double val = sc_twoe(h2e, r1, a1, r2, a2, n_occ, n_in) - sc_twoe(h2e, r1, a2, r2, a1, n_occ, n_in);
                size_t s = SBj;
                int p1 = phase_single_excitation(a1, r1, s);
                s = (s & ~(1UL << a1)) | (1UL << r1);
                int p2 = phase_single_excitation(a2, r2, s);
                h = (p1 * p2) * val;
            } else { /* da==1, db==1 : one alpha, one beta (opposite spin, no exchange) */
                int a = __builtin_ctzll(SAj & ~SAi);
                int r = __builtin_ctzll(SAi & ~SAj);
                int b = __builtin_ctzll(SBj & ~SBi);
                int s = __builtin_ctzll(SBi & ~SBj);
                double val = sc_twoe(h2e, r, a, s, b, n_occ, n_in);
                int pa = phase_single_excitation(a, r, SAj);
                int pb = phase_single_excitation(b, s, SBj);
                h = (pa * pb) * val;
            }
        }
    } else if (dm == 1 || dm == -1) {
        /* ---- dipole coupling m <-> m+-1 (one-electron operator) ---- */
        int mmax = mi > mj ? mi : mj;
        double pref = -sqrt(mmax * omega / 2.0);
        int diff = da + db;
        if (diff == 0) {
            double dsum = 0.0;
            for (int x = 0; x < nAi; x++) { int p = Ai[x]; dsum += d_cmo[(p + n_in) * (size_t)n_occ + (p + n_in)]; }
            for (int x = 0; x < nBi; x++) { int p = Bi[x]; dsum += d_cmo[(p + n_in) * (size_t)n_occ + (p + n_in)]; }
            h = pref * (dsum - d_exp);
        } else if (diff == 1) {
            int a, r, alpha;
            if (da == 1) { a = __builtin_ctzll(SAj & ~SAi); r = __builtin_ctzll(SAi & ~SAj); alpha = 1; }
            else         { a = __builtin_ctzll(SBj & ~SBi); r = __builtin_ctzll(SBi & ~SBj); alpha = 0; }
            int ph = phase_single_excitation(a, r, alpha ? SAj : SBj);
            h = pref * ph * d_cmo[(r + n_in) * (size_t)n_occ + (a + n_in)];
        }
    }
    return h;
}

void build_guess_blocks_scrules(double* G_out, double* Gs_out, const int* sel, int L,
        const double* h1e, const double* h2e, const double* d_cmo, const double* Sdiag,
        int N_ac, int n_o_ac, int n_o_in, int N_p, int num_alpha,
        double Enuc, double dc, double omega, double d_exp, double E_core) {
    int n_occ = n_o_ac + n_o_in;
    int n_in = n_o_in;
    (void)N_p;
    size_t num_dets = (size_t)num_alpha * num_alpha;

    /* graph for index -> string (rebuilt locally so davidson_spin's signature
       is untouched; same call get_string uses) */
    int grows = N_ac * (n_o_ac - N_ac + 1);
    int* Y = (int*) malloc((size_t)grows * 3 * sizeof(int));
    get_graph(N_ac, n_o_ac, Y);

    /* decode each selected determinant once */
    int*    mm  = (int*)    malloc(L * sizeof(int));
    size_t* Idt = (size_t*) malloc(L * sizeof(size_t));
    size_t* SA  = (size_t*) malloc(L * sizeof(size_t));
    size_t* SB  = (size_t*) malloc(L * sizeof(size_t));
    int**   AA  = (int**)   malloc(L * sizeof(int*));
    int**   BB  = (int**)   malloc(L * sizeof(int*));
    int*    nA  = (int*)    malloc(L * sizeof(int));
    int*    nB  = (int*)    malloc(L * sizeof(int));
    for (int i = 0; i < L; i++) {
        size_t d = (size_t) sel[i];
        size_t Idet = d % num_dets;
        Idt[i] = Idet;
        mm[i]  = (int)((d - Idet) / num_dets);
        int ib = (int)(Idet % num_alpha);
        int ia = (int)((Idet - ib) / num_alpha);
        SA[i] = index_to_string(ia, N_ac, n_o_ac, Y);
        SB[i] = index_to_string(ib, N_ac, n_o_ac, Y);
        AA[i] = string_to_obtlist(SA[i], n_o_ac, &nA[i]);
        BB[i] = string_to_obtlist(SB[i], n_o_ac, &nB[i]);
    }

    /* build_sigma's h1e is the E-operator "modified" one-electron integral
       k_ps = F^c_ps - 0.5 * sum_r (pr|rs)  (the -1/2 delta_qr E_ps piece of
       H = sum k E + 1/2 (pq|rs) E_pq E_rs folded into the 1e).  Rebuild the
       physical active core-Fock F^c so the standard determinant Slater-Condon
       formulas below apply without double-counting exchange. */
    double* hp = (double*) malloc((size_t)n_o_ac * n_o_ac * sizeof(double));
    for (int p = 0; p < n_o_ac; p++)
        for (int s = 0; s < n_o_ac; s++) {
            double v = h1e[p * n_o_ac + s];
            for (int r = 0; r < n_o_ac; r++) v += 0.5 * sc_twoe(h2e, p, r, r, s, n_occ, n_in);
            hp[p * n_o_ac + s] = v;
        }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            /* H comes from the shared Slater-Condon helper; S^2 stays here
             * because only the guess block needs it. */
            double h = sc_h_element(SA[i], SB[i], AA[i], nA[i], BB[i], nB[i], mm[i],
                                    SA[j], SB[j], mm[j],
                                    hp, h2e, d_cmo, n_o_ac, n_occ, n_in,
                                    Enuc, dc, omega, d_exp, E_core);
            double s2 = 0.0;
            if (mm[i] == mm[j]) {
                /* ---- S^2_00 (block-diagonal in photon) ---- */
                if (i == j) {
                    s2 = Sdiag[Idt[i]];
                } else if (__builtin_popcountll(SA[i] ^ SA[j]) / 2 == 1 &&
                           __builtin_popcountll(SB[i] ^ SB[j]) / 2 == 1) {
                    /* spin-flip: alpha p->q must mirror beta q->p */
                    int a = __builtin_ctzll(SA[j] & ~SA[i]); /* alpha hole  (in ket) */
                    int r = __builtin_ctzll(SA[i] & ~SA[j]); /* alpha particle (bra) */
                    int b = __builtin_ctzll(SB[j] & ~SB[i]); /* beta  hole  (in ket) */
                    int s = __builtin_ctzll(SB[i] & ~SB[j]); /* beta  particle (bra) */
                    if (r == b && s == a) {
                        int pa = phase_single_excitation(a, r, SA[j]);
                        int pb = phase_single_excitation(b, s, SB[j]);
                        s2 = -(double)(pa * pb);
                    }
                }
            }

            G_out[(size_t)i * L + j]  = h;
            Gs_out[(size_t)i * L + j] = s2;
        }
    }
    for (int i = 0; i < L; i++) { free(AA[i]); free(BB[i]); }
    free(mm); free(Idt); free(SA); free(SB); free(AA); free(BB); free(nA); free(nB);
    free(hp);
    free(Y);
}

/* Every string reachable from s0 by exactly `level` (1 or 2) excitations, as
 * determinant-string indices.  Used by build_sparse_init_sigma to walk a
 * column's connected set directly instead of testing it against all H_dim
 * determinants.  Singles and doubles are disjoint and duplicate-free (they
 * differ from s0 in 2 and 4 bit positions respectively), which is what lets
 * the caller union the buckets without double counting. */
static int enum_excited_indices(size_t s0, int level, int N_ac, int n_o_ac, int* Y, int* out) {
    int occ[64], vir[64], no = 0, nv = 0;
    for (int p = 0; p < n_o_ac; p++) {
        if (s0 & (1UL << p)) occ[no++] = p; else vir[nv++] = p;
    }
    int c = 0;
    if (level == 1) {
        for (int i = 0; i < no; i++)
            for (int a = 0; a < nv; a++) {
                size_t s = (s0 & ~(1UL << occ[i])) | (1UL << vir[a]);
                out[c++] = string_to_index(s, N_ac, n_o_ac, Y);
            }
    } else {
        for (int i = 0; i < no; i++)
            for (int j = i + 1; j < no; j++)
                for (int a = 0; a < nv; a++)
                    for (int b = a + 1; b < nv; b++) {
                        size_t s = (s0 & ~(1UL << occ[i]) & ~(1UL << occ[j]))
                                 | (1UL << vir[a]) | (1UL << vir[b]);
                        out[c++] = string_to_index(s, N_ac, n_o_ac, Y);
                    }
    }
    return c;
}

/* Direct Slater-Condon build of S += H*Q for trial vectors whose support is
 * confined to a small set of determinants.  Enabled by QED_SPARSE_INIT_SIGMA.
 *
 * WHY THIS EXISTS.  build_sigma is STRICTLY LINEAR in the number of trial
 * vectors -- it loops `for n in 0..num_state` doing one full, independent
 * sigma product per vector, with no batching or amortization across them.
 * That is the right thing for the dense vectors the Davidson iteration
 * generates, but the FIRST iteration (and every restart) starts from the guess
 * set, whose vectors live entirely inside the L selected determinants:
 *   - plain davidson     : unit vectors, support 1 each;
 *   - davidson_spin      : the spin-adapted combinations built at the end of
 *                          the guess stage, support inside the same L.
 * For those, H*Q is a handful of SPARSE COLUMNS of H rather than num_state
 * dense sigma products.  On MgH+ CAS(8,12) (H_dim 245025) each column has only
 * ~1425 nonzeros (0.58%%), and that first build measured 26.6 s of a 88 s CI
 * solve.
 *
 * METHOD.  Pair testing, not excitation enumeration: for every determinant J
 * we test it against each support column with two XOR/popcount pairs and let
 * sc_h_element reject at >2 excitations.  That is H_dim*ncol cheap tests --
 * more than the ~ncol*1425 an enumerating build would touch, but it reuses the
 * already-validated Slater-Condon path verbatim and needs no excitation
 * generator or inverse addressing.  Only the alpha/beta STRINGS are decoded
 * (num_alpha of them, not H_dim), so the hot loop stays in cache.
 *
 * RETURNS 1 if it built S, 0 if it DECLINED because the vectors are not sparse
 * enough -- the caller must then fall back to build_sigma.  The acceptance rule
 * is ncol <= 8*num_state, which bounds the work at 8x the size of the output
 * that has to be written anyway; dense Davidson vectors always decline.
 *
 * Accumulates (+=) exactly like build_sigma, so callers that memset S first
 * get identical semantics. */
int build_sparse_init_sigma(const double* h1e, const double* h2e, const double* d_cmo,
        const double* Q, double* S, int num_state,
        int N_ac, int n_o_ac, int n_o_in, int N_p, int num_alpha,
        double Enuc, double dc, double omega, double d_exp, double E_core,
        const double* Sdiag, double s2_scale) {
    int n_occ = n_o_ac + n_o_in;
    int n_in  = n_o_in;
    int np1   = N_p + 1;
    size_t num_dets = (size_t) num_alpha * num_alpha;
    size_t H_dim    = (size_t) np1 * num_dets;
    if (num_state <= 0) return 1;
    /* s2_scale != 0 folds the S^2 SPIN PENALTY into the same operator, so the
     * caller must then SKIP its separate build_sigma_s_square call.  This is
     * free to do here: S^2 is nonzero only on the diagonal and on the
     * da==1 && db==1 spin-flip pairs, both of which are already inside the
     * <=2-excitation set enumerated below, and build_sigma_s_square is itself
     * strictly linear in the vector count (one full S^2 pass per vector),
     * which is what made it 2.05 s of a 2.42 s iteration 1 on CAS(8,12). */

    /* ---- union support of the trial vectors ------------------------------
     * Scanned state-major so the reads are contiguous.  Bails out as soon as
     * the support exceeds the cap, so declining is cheap. */
    double _t0 = omp_get_wtime();
    size_t cap = (size_t) 8 * (size_t) num_state;
    unsigned char* mark = (unsigned char*) calloc(H_dim, 1);
    if (!mark) return 0;
    size_t ncol = 0;
    for (int v = 0; v < num_state && ncol <= cap; v++) {
        const double* q = Q + (size_t) v * H_dim;
        for (size_t d = 0; d < H_dim; d++)
            if (q[d] != 0.0 && !mark[d]) { mark[d] = 1; ncol++; }
    }
    if (ncol > cap) { free(mark); return 0; }
    if (ncol == 0)  { free(mark); return 1; }   /* Q is all zero: S += 0 */

    int* col = (int*) malloc(ncol * sizeof(int));
    if (!col) { free(mark); return 0; }
    { size_t c = 0; for (size_t d = 0; d < H_dim; d++) if (mark[d]) col[c++] = (int) d; }
    free(mark);

    double _t1 = omp_get_wtime();
    /* ---- decode the num_alpha distinct strings once (NOT H_dim of them) --- */
    int grows = N_ac * (n_o_ac - N_ac + 1);
    int* Y = (int*) malloc((size_t) grows * 3 * sizeof(int));
    get_graph(N_ac, n_o_ac, Y);
    size_t* SS = (size_t*) malloc((size_t) num_alpha * sizeof(size_t));
    int**   OL = (int**)   malloc((size_t) num_alpha * sizeof(int*));
    int*    NL = (int*)    malloc((size_t) num_alpha * sizeof(int));
    for (int ia = 0; ia < num_alpha; ia++) {
        SS[ia] = index_to_string(ia, N_ac, n_o_ac, Y);
        OL[ia] = string_to_obtlist(SS[ia], n_o_ac, &NL[ia]);
    }

    /* Physical active core-Fock from build_sigma's modified h1e -- identical
     * to build_guess_blocks_scrules, and required for the same reason (else
     * the exchange term is double counted). */
    double* hp = (double*) malloc((size_t) n_o_ac * n_o_ac * sizeof(double));
    for (int p = 0; p < n_o_ac; p++)
        for (int s = 0; s < n_o_ac; s++) {
            double v = h1e[p * n_o_ac + s];
            for (int r = 0; r < n_o_ac; r++) v += 0.5 * sc_twoe(h2e, p, r, r, s, n_occ, n_in);
            hp[p * n_o_ac + s] = v;
        }

    /* ---- decode the support columns and gather their coefficients --------- */
    int* mk  = (int*) malloc(ncol * sizeof(int));
    int* iak = (int*) malloc(ncol * sizeof(int));
    int* ibk = (int*) malloc(ncol * sizeof(int));
    double* Cmat = (double*) malloc(ncol * (size_t) num_state * sizeof(double));
    for (size_t c = 0; c < ncol; c++) {
        size_t d = (size_t) col[c];
        size_t Idet = d % num_dets;
        mk[c]  = (int) (d / num_dets);
        ibk[c] = (int) (Idet % num_alpha);
        iak[c] = (int) (Idet / num_alpha);
        for (int v = 0; v < num_state; v++)
            Cmat[c * (size_t) num_state + v] = Q[(size_t) v * H_dim + d];
    }

    double _t2 = omp_get_wtime();
    /* ---- phase 1: enumerate each column's connected determinants ---------
     * For column K only the determinants within 2 excitations can contribute,
     * so walk them directly rather than testing K against all H_dim.  On
     * CAS(8,12) that is 1425 per column instead of 245025 -- the whole point
     * of enumeration over pair testing.
     *
     * The (da, db) buckets below partition the connected set: every J has one
     * definite alpha and beta excitation level, so each is generated EXACTLY
     * ONCE and the contributions must not be de-duplicated.  dm = +-1 (the
     * dipole coupling) is restricted to da+db <= 1 because the dipole is a
     * one-electron operator; sc_h_element returns 0 beyond that anyway. */
    int nsa_max = N_ac * (n_o_ac - N_ac);
    int nda_max = (N_ac * (N_ac - 1) / 2) * ((n_o_ac - N_ac) * (n_o_ac - N_ac - 1) / 2);
    size_t nconn = (size_t) (1 + nsa_max + nda_max)
                 + (size_t) nsa_max * (1 + nsa_max)
                 + (size_t) nda_max;
    size_t maxper = (size_t) (np1 < 3 ? np1 : 3) * nconn;

    size_t* eJ = (size_t*) malloc(ncol * maxper * sizeof(size_t));
    double* eH = (double*) malloc(ncol * maxper * sizeof(double));
    int*    en = (int*)    malloc(ncol * sizeof(int));
    if (!eJ || !eH || !en) {
        free(eJ); free(eH); free(en);
        for (int ia = 0; ia < num_alpha; ia++) free(OL[ia]);
        free(SS); free(OL); free(NL); free(Y); free(hp);
        free(mk); free(iak); free(ibk); free(Cmat); free(col);
        return 0;   /* let the caller fall back rather than die on a big case */
    }

    #pragma omp parallel
    {
        int* a1 = (int*) malloc((nsa_max ? nsa_max : 1) * sizeof(int));
        int* a2 = (int*) malloc((nda_max ? nda_max : 1) * sizeof(int));
        int* b1 = (int*) malloc((nsa_max ? nsa_max : 1) * sizeof(int));
        int* b2 = (int*) malloc((nda_max ? nda_max : 1) * sizeof(int));
        #pragma omp for schedule(dynamic)
        for (long long cc = 0; cc < (long long) ncol; cc++) {
            size_t c = (size_t) cc;
            int iaK = iak[c], ibK = ibk[c], mK = mk[c];
            int na1 = enum_excited_indices(SS[iaK], 1, N_ac, n_o_ac, Y, a1);
            int na2 = enum_excited_indices(SS[iaK], 2, N_ac, n_o_ac, Y, a2);
            int nb1 = enum_excited_indices(SS[ibK], 1, N_ac, n_o_ac, Y, b1);
            int nb2 = enum_excited_indices(SS[ibK], 2, N_ac, n_o_ac, Y, b2);
            size_t* Jout = eJ + c * maxper;
            double* Hout = eH + c * maxper;
            int cnt = 0;
            for (int da = 0; da <= 2; da++) {
                int nia = (da == 0) ? 1 : ((da == 1) ? na1 : na2);
                for (int x = 0; x < nia; x++) {
                    int iaJ = (da == 0) ? iaK : ((da == 1) ? a1[x] : a2[x]);
                    for (int db = 0; db <= 2 - da; db++) {
                        int nib = (db == 0) ? 1 : ((db == 1) ? nb1 : nb2);
                        for (int y = 0; y < nib; y++) {
                            int ibJ = (db == 0) ? ibK : ((db == 1) ? b1[y] : b2[y]);
                            for (int dmv = -1; dmv <= 1; dmv++) {
                                if (dmv != 0 && da + db > 1) continue;
                                int mJ = mK + dmv;
                                if (mJ < 0 || mJ > N_p) continue;
                                size_t Jd = (size_t) mJ * num_dets
                                          + (size_t) iaJ * num_alpha + (size_t) ibJ;
                                double h = sc_h_element(SS[iaJ], SS[ibJ], OL[iaJ], NL[iaJ],
                                                        OL[ibJ], NL[ibJ], mJ,
                                                        SS[iaK], SS[ibK], mK,
                                                        hp, h2e, d_cmo, n_o_ac, n_occ, n_in,
                                                        Enuc, dc, omega, d_exp, E_core);
                                /* S^2 penalty, same element convention as
                                 * build_guess_blocks_scrules (bra = J, ket = K;
                                 * phases evaluated on the ket's strings).
                                 * Block diagonal in the photon index. */
                                if (s2_scale != 0.0 && dmv == 0) {
                                    if (da == 0 && db == 0) {
                                        h += s2_scale * Sdiag[(size_t) iaJ * num_alpha + ibJ];
                                    } else if (da == 1 && db == 1) {
                                        size_t SAk = SS[iaK], SBk = SS[ibK];
                                        size_t SAj = SS[iaJ], SBj = SS[ibJ];
                                        int aa = __builtin_ctzll(SAk & ~SAj);
                                        int rr = __builtin_ctzll(SAj & ~SAk);
                                        int bb = __builtin_ctzll(SBk & ~SBj);
                                        int ss = __builtin_ctzll(SBj & ~SBk);
                                        if (rr == bb && ss == aa) {
                                            int pa = phase_single_excitation(aa, rr, SAk);
                                            int pb = phase_single_excitation(bb, ss, SBk);
                                            h -= s2_scale * (double) (pa * pb);
                                        }
                                    }
                                }
                                if (h == 0.0) continue;
                                Jout[cnt] = Jd;
                                Hout[cnt] = h;
                                cnt++;
                            }
                        }
                    }
                }
            }
            en[c] = cnt;
        }
        free(a1); free(a2); free(b1); free(b2);
    }

    double _t3 = omp_get_wtime();
    /* ---- phase 2: invert to CSR keyed by J -------------------------------
     * Scattering straight from the columns would race (several columns hit the
     * same J).  Inverting once costs one pass over ~ncol*nconn entries and
     * lets phase 3 stay parallel over J with no atomics. */
    size_t* rowp = (size_t*) calloc(H_dim + 1, sizeof(size_t));
    size_t nnz = 0;
    for (size_t c = 0; c < ncol; c++) {
        const size_t* Jout = eJ + c * maxper;
        for (int k = 0; k < en[c]; k++) rowp[Jout[k] + 1]++;
        nnz += (size_t) en[c];
    }
    for (size_t J = 0; J < H_dim; J++) rowp[J + 1] += rowp[J];
    int*    ecol = (int*)    malloc((nnz ? nnz : 1) * sizeof(int));
    double* eval = (double*) malloc((nnz ? nnz : 1) * sizeof(double));
    size_t* pos  = (size_t*) malloc((H_dim + 1) * sizeof(size_t));
    memcpy(pos, rowp, (H_dim + 1) * sizeof(size_t));
    for (size_t c = 0; c < ncol; c++) {
        const size_t* Jout = eJ + c * maxper;
        const double* Hout = eH + c * maxper;
        for (int k = 0; k < en[c]; k++) {
            size_t p = pos[Jout[k]]++;
            ecol[p] = (int) c;
            eval[p] = Hout[k];
        }
    }
    free(pos); free(eJ); free(eH); free(en);

    double _t4 = omp_get_wtime();
    /* ---- phase 3: S[v][J] += sum_c H[J, col[c]] * Q[v][col[c]] ------------
     * Parallel over BLOCKS of J: each thread owns distinct J values and writes
     * only S[*][J], so there is no race and no reduction.  Accumulating into a
     * block-local T indexed [jj][v] keeps the inner v loop contiguous, and the
     * transpose back into S is one contiguous run per state. */
    long long Hll = (long long) H_dim;
    const int BJ = 256;
    #pragma omp parallel
    {
        double* T = (double*) malloc((size_t) BJ * num_state * sizeof(double));
        #pragma omp for schedule(static)
        for (long long J0 = 0; J0 < Hll; J0 += BJ) {
            int nb = (int) ((J0 + BJ <= Hll) ? BJ : (Hll - J0));
            memset(T, 0, (size_t) nb * num_state * sizeof(double));
            for (int jj = 0; jj < nb; jj++) {
                size_t Jd = (size_t) (J0 + jj);
                double* t = T + (size_t) jj * num_state;
                for (size_t e = rowp[Jd]; e < rowp[Jd + 1]; e++) {
                    double h = eval[e];
                    const double* cst = Cmat + (size_t) ecol[e] * (size_t) num_state;
                    for (int v = 0; v < num_state; v++) t[v] += h * cst[v];
                }
            }
            for (int v = 0; v < num_state; v++) {
                double* srow = S + (size_t) v * H_dim + (size_t) J0;
                for (int jj = 0; jj < nb; jj++) srow[jj] += T[(size_t) jj * num_state + v];
            }
        }
        free(T);
    }
    free(rowp); free(ecol); free(eval);
    if (get_ci_print_level() >= 3) { double _t5 = omp_get_wtime();
      printf("[sparse-prof] ncol=%zu nnz=%zu num_state=%d H_dim=%zu | scan %.3f  decode %.3f  ph1 %.3f  ph2 %.3f  ph3 %.3f  TOTAL %.3f\n",
        ncol, nnz, num_state, H_dim, _t1-_t0, _t2-_t1, _t3-_t2, _t4-_t3, _t5-_t4, _t5-_t0); fflush(stdout); }

    for (int ia = 0; ia < num_alpha; ia++) free(OL[ia]);
    free(SS); free(OL); free(NL); free(Y); free(hp);
    free(mk); free(iak); free(ibk); free(Cmat); free(col);
    return 1;
}

/* QED_SPARSE_INIT_COMPARE: rebuild the same S through build_sparse_init_sigma
 * into scratch and report the largest disagreement with the S that build_sigma
 * just produced.  Purely diagnostic -- the caller KEEPS build_sigma's result,
 * so enabling the compare never changes a run's numbers (unlike enabling
 * QED_SPARSE_INIT_SIGMA itself, which changes the summation order). */
static void sparse_init_sigma_compare(const double* h1e, const double* h2e, const double* d_cmo,
        const double* Q, const double* S, int num_state,
        int N_ac, int n_o_ac, int n_o_in, int N_p, int num_alpha,
        double Enuc, double dc, double omega, double d_exp, double E_core,
        const double* Sdiag, double s2_scale, size_t H_dim) {
    size_t n = (size_t) num_state * H_dim;
    double* Sc = (double*) calloc(n, sizeof(double));
    if (!Sc) {
        printf("[sparse-init-sigma] compare skipped: could not allocate %.1f MiB\n", n * 8.0 / 1048576.0);
        fflush(stdout);
        return;
    }
    if (!build_sparse_init_sigma(h1e, h2e, d_cmo, Q, Sc, num_state,
                N_ac, n_o_ac, n_o_in, N_p, num_alpha, Enuc, dc, omega, d_exp, E_core,
                Sdiag, s2_scale)) {
        printf("[sparse-init-sigma] compare skipped: declined (guess vectors not sparse enough)\n");
        fflush(stdout);
        free(Sc);
        return;
    }
    /* rows 0..num_state-1 of the caller's S are contiguous (S is maxdim x H_dim) */
    double mx = 0.0, mref = 0.0;
    for (size_t t = 0; t < n; t++) {
        double d = fabs(Sc[t] - S[t]);
        if (d > mx) mx = d;
        double a = fabs(S[t]);
        if (a > mref) mref = a;
    }
    printf("[sparse-init-sigma] num_state=%d  max|dS|=%.3e  max|S|=%.3e  rel=%.3e\n",
           num_state, mx, mref, mref > 0.0 ? mx / mref : 0.0);
    fflush(stdout);
    free(Sc);
}

void davidson_spin(double* h1e, double* h2e, double* d_cmo, double* Hdiag, double* Sdiag, double* Sdiag_projection, double* eigenvals, double* eigenvecs, int* table,
		int* table_creation, int* table_annihilation, int* b_array, int *constint, double *constdouble, int* index_Hdiag,
		bool casscf, double target_spin, callback_ build_sigma) {
    //unpack constant

    int N_ac = constint[0]; 
    int n_o_ac = constint[1]; 
    int n_o_in = constint[2]; 
    int nmo = constint[3];    
    int N_p = constint[4];   
    int indim = constint[5];         
    int maxdim = constint[6];
    int nroots = constint[7];
    int maxiter = constint[8];
    
    double Enuc = constdouble[0]; 
    double dc = constdouble[1]; 
    double omega = constdouble[2]; 
    double d_exp = constdouble[3];
    double threshold = constdouble[4];
    double E_core = constdouble[5];
    bool break_degeneracy = false;         
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    size_t H_dim = (N_p + 1) * num_alpha * num_alpha;
    /* Trace-only per-iteration/diagnostic output (see get_ci_print_level).
     * At Normal/Debug the CI solver stays quiet apart from per-solve
     * convergence and (level-gated) spin reporting. */
    int trace = (get_ci_print_level() >= 3);
    if (trace) printf("target_spin %20.12lf\n",target_spin);
    if (trace) printf(" indim%d", indim);
    const int indim_requested = indim;   /* davidson_indim * nroots, before growth */
    /* QED_SPARSE_INIT_SIGMA: build the first/restart sigma directly from the
     * sparse guess vectors instead of one full sigma product per vector (see
     * build_sparse_init_sigma).  QED_SPARSE_INIT_COMPARE builds it both ways
     * and reports the discrepancy while keeping build_sigma's result. */
    int sparse_init = 0; { const char* e = getenv("QED_SPARSE_INIT_SIGMA");   if (e && atoi(e) != 0) sparse_init = 1; }
    int sparse_cmp  = 0; { const char* e = getenv("QED_SPARSE_INIT_COMPARE"); if (e && atoi(e) != 0) sparse_cmp  = 1; }

    /* Neither subspace size can exceed the CI space itself.  helper_PFCI.py:3076
     * checks this before calling, but the C backend is also driven directly
     * (cpp_casscf), so enforce it here where the buffers are actually sized. */
    if ((size_t) indim_requested > H_dim || (size_t) maxdim > H_dim) {
        printf(
"\n********************************************************************************\n"
" CI Davidson cannot start: requested subspace is larger than the CI space.\n\n"
"   davidson_indim  * nroots = %d\n"
"   davidson_maxdim * nroots = %d\n"
"   CI dimension H_dim       = %zu   <-- neither may exceed this\n"
"   nroots = %d\n\n"
" NOTE: davidson_indim and davidson_maxdim are PER-ROOT MULTIPLIERS, so the\n"
" sizes that matter are the products above, not the values you set.\n\n"
" FIX: set davidson_indim and davidson_maxdim to at most %zu (= H_dim / nroots),\n"
" keeping davidson_maxdim > davidson_indim so the guess set has room to grow.\n"
" For a CI space this small, full diagonalization is usually the better choice.\n"
"********************************************************************************\n",
            indim_requested, maxdim, H_dim, nroots,
            nroots ? H_dim / (size_t) nroots : H_dim);
        fflush(stdout);
        exit(1);
    }
    /* QED_HYBRID_GUESS = N -- the developer's hybrid guess (item 218C): keep the
     * warm CI vector(s) AND add an N-determinant P-space of unit vectors chosen
     * from the CURRENT H_diag, spin-adapted through the same Algorithm 3 path
     * the cold start uses, then orthonormalized against the warm vector(s).
     * Default 0 = off, so this is a pure addition to the existing two paths.
     *
     * Only meaningful when a warm vector exists (casscf == true).  With
     * casscf == false the cold path already builds exactly this P-space and
     * there is no warm vector to add, so the knob is ignored there.
     *
     * NOTE ON index_Hdiag: the closure below walks the current Hdiag in
     * index_Hdiag's order, and on the micro path index_Hdiag is the SETUP
     * argsort unless QED_FRESH_HDIAG_SORT=1 (helper_PFCI.py).  The hybrid
     * therefore wants that knob on; see the warning printed below. */
    int hybrid_n = 0; { const char* e = getenv("QED_HYBRID_GUESS"); if (e) hybrid_n = atoi(e); }
    int hybrid = (hybrid_n > 0) && (casscf == true);
    int hybrid_check = 0; { const char* e = getenv("QED_HYBRID_CHECK"); if (e && atoi(e) != 0) hybrid_check = 1; }
    /* Number of trailing rows of the assembled guess that are DENSE (the warm
     * vectors).  0 on both non-hybrid paths.  Read by the sparse/dense split at
     * the first sigma build. */
    int hybrid_dense_tail = 0;
    if (casscf == true && !hybrid) {
        //maxiter = 3;
        indim = nroots;
    }
    else {
    //getting all configurations in the spin coupling set
        if (hybrid) {
            /* Hazard 4 of item 218C: `indim = nroots` is set in the OTHER
             * branch, so the hybrid has to establish its own P-space size here
             * before the shared closure runs. */
            indim = hybrid_n;
            if (indim > (int) H_dim) indim = (int) H_dim;
        }
        /* The bound test MUST come first: at indim == H_dim the other operand
         * evaluates Hdiag[index_Hdiag[H_dim]], one element past the end of
         * index_Hdiag.  && short-circuits left to right, so ordering it this
         * way is what makes the read safe. */
        while((indim < H_dim) && (Hdiag[index_Hdiag[indim]] - Hdiag[index_Hdiag[indim-1]] < 1e-10)) {
            indim += 1;
        }
        if (hybrid && indim + nroots > maxdim) {
            /* Hazard 2 of item 218C, with the hybrid's extra nroots rows: the
             * warm vectors sit in Q alongside the closed P-space, so the buffer
             * has to hold both.  Same reasoning as the cold-path check below --
             * refuse rather than overflow the heap, and do not truncate (a cut
             * spin-coupling set contaminates the spin-filtered guess). */
            printf(
"\n********************************************************************************\n"
" CI Davidson (hybrid guess) cannot start: warm vectors + P-space do not fit.\n\n"
"   QED_HYBRID_GUESS (requested P-space)        = %d\n"
"   after closing the spin-coupling set         = %d\n"
"   plus warm vectors (nroots)                  = %d\n"
"   total needed                                = %d   <-- must fit below\n"
"   davidson_maxdim * nroots (buffer capacity)  = %d\n\n"
" FIX: raise davidson_maxdim, or lower QED_HYBRID_GUESS.\n"
"********************************************************************************\n",
                hybrid_n, indim, nroots, indim + nroots, maxdim);
            fflush(stdout);
            exit(1);
        }
        /* The growth above is data dependent (it follows the H_diag degeneracy
         * pattern, which moves as the orbitals move), so the selection set can
         * end up larger than the davidson_indim*nroots that was asked for.
         * Q, S and theta are all sized by maxdim (= davidson_maxdim*nroots) but
         * are written with `indim` entries below, so indim > maxdim is a heap
         * overflow.  It corrupts memory silently and surfaces much later and
         * far away as an all-NaN CI vector, so refuse here instead.
         * Truncating is NOT an alternative: a partial spin-coupling set leaves
         * S^2 non-block-diagonal in the selected space and the spin-filtered
         * guess is then contaminated. */
        if (indim > maxdim) {
            printf(
"\n********************************************************************************\n"
" CI Davidson (spin-adapted) cannot start: the initial determinant set does not\n"
" fit inside the subspace buffers.\n\n"
"   davidson_indim * nroots  (requested)        = %d\n"
"   after closing the spin-coupling set         = %d   <-- must fit below\n"
"   davidson_maxdim * nroots (buffer capacity)  = %d\n"
"   CI dimension H_dim                          = %zu\n"
"   nroots = %d, target spin S = %.1f\n\n"
" WHAT HAPPENED.  The guess is built from the lowest-energy determinants.  That\n"
" set is then GROWN until it no longer splits a group of degenerate\n"
" determinants, so that every spin-coupling set it touches is complete -- if it\n"
" were cut, S^2 would not be block diagonal over the selected determinants and\n"
" the spin-filtered guess would be contaminated.  How far it has to grow depends\n"
" on the orbitals, so it changes from one macroiteration to the next and can\n"
" exceed the size you asked for.\n\n"
" NOTE: davidson_indim and davidson_maxdim are PER-ROOT MULTIPLIERS.  With\n"
" nroots = %d, davidson_indim = %d selects %d determinants, not %d.\n\n"
" FIX: raise davidson_maxdim (or lower davidson_indim) so the grown set fits.\n"
" davidson_maxdim >= davidson_indim + 2 is usually enough headroom.  Setting\n"
" davidson_indim == davidson_maxdim leaves NO room for the growth and will hit\n"
" this every time the guess set touches a degeneracy.  For a CI space this\n"
" small, full diagonalization is also a reasonable option.\n"
"********************************************************************************\n",
                indim_requested, indim, maxdim, H_dim, nroots, target_spin,
                nroots, indim_requested / (nroots ? nroots : 1),
                indim_requested, indim_requested / (nroots ? nroots : 1));
            fflush(stdout);
            exit(1);
        }
    }
    if (trace) printf(" maxiter%d", maxiter);


    int num_links0 = N_ac*(n_o_ac-N_ac)+N_ac;
    //for (int i = 0; i < H_dim; i++) {
    //    //printf("%4d\n", index_Hdiag[i]);    
    //    printf("%4d %20.12lf\n", index_Hdiag[i], Hdiag[index_Hdiag[i]]);    
    //}
    
    int L = indim;
    if (trace) printf(" indim%d H_dim %zu\n", indim, H_dim);
    double* Hdiag2 = (double*)malloc(H_dim*sizeof(double));
    cblas_dcopy(H_dim,Hdiag,1,Hdiag2,1);
    double* Q = (double*) malloc(maxdim*H_dim*sizeof(double));
    /* Only the rows that are actually part of the subspace need zeroing.
    * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
    * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
    * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
    * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
    * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
    * where L is 181.
    * Here: L == indim here; the unit-vector guess below writes ONE element per row, so those rows do need the zeros. */
    memset(Q, 0, (size_t) L*H_dim*sizeof(double));
    double* S = (double*) malloc(maxdim*H_dim*sizeof(double));

    double* w = (double*) malloc(nroots*H_dim*sizeof(double));
    int* unconverged_idx = (int*) malloc(nroots*sizeof(int));
    bool* convergence_check = (bool*) malloc(nroots*sizeof(bool));
    double* theta = (double*) malloc(maxdim*sizeof(double));
    memset(theta, 0, maxdim*sizeof(double));
    /* [step7] snapshot of the CSF-basis eigenvalues e from Algorithm 3 step 6,
     * kept so iteration 1 can check that the projected subspace Hamiltonian is
     * diag(e) -- which it is if and only if step 7 is transposed correctly.
     * NULL on the warm (casscf == true) path, which has no guess block. */
    double* guess_theta = NULL;
    int guess_theta_n = 0;
        
    double minimum = 0.0;
    int min_pos;
    if (casscf == false || hybrid) {
        /* The hybrid runs this whole P-space build unchanged -- selection,
         * Algorithm 3 guess blocks, S^2 filter, steps 7 and 8 -- and differs
         * only in what is done with the result at the end of the block, where
         * the warm vector(s) are prepended and the set is re-orthonormalized. */
        //use unit vectors as guess
        int* sel = (int*) malloc(indim * sizeof(int)); /* selected determinant indices (Algorithm 3 direct block) */
        for (int i = 0; i < indim; i++) {
            minimum = Hdiag2[0];
            min_pos = 0;
            for (int j = 1; j < H_dim; j++){
                if (Hdiag2[j] < minimum) {
                    minimum = Hdiag2[j];
                    min_pos = j;
                }
            }
            Q[i * H_dim + min_pos] = 1.0;
            sel[i] = min_pos;
            Hdiag2[min_pos] = BIGNUM;
           // lambda_oldp[i] = minimum;
        }

        /* CI guess blocks H_00 (G) and S^2_00 (Gs), Algorithm 3.
         * Default: direct Slater-Condon build (build_guess_blocks_scrules --
         * touches only the L selected determinants).  QED_GUESS_SIGMA=1
         * restores the legacy build via L full-space sigma-vector products.
         * QED_GUESS_COMPARE=1 builds both and reports the max discrepancy. */
        int guess_sigma = 0; { const char* e = getenv("QED_GUESS_SIGMA");   if (e && atoi(e) != 0) guess_sigma = 1; }
        int guess_cmp   = 0; { const char* e = getenv("QED_GUESS_COMPARE"); if (e && atoi(e) != 0) guess_cmp   = 1; }

        double* G    = (double*) malloc(L*L*sizeof(double));
        memset(G, 0, L*L*sizeof(double));
        double* newS = (double*) malloc(L*H_dim*sizeof(double));
        memset(newS, 0, L*H_dim*sizeof(double));
        double* Gs   = (double*) malloc(L*L*sizeof(double));
        memset(Gs, 0, L*L*sizeof(double));

        if (guess_sigma || guess_cmp) {
            /* legacy: L full-space sigma-vector products, projected to the L x L block */
            /* Only the rows that are actually part of the subspace need zeroing.
            * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
            * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
            * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
            * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
            * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
            * where L is 181.
            * Here: legacy QED_GUESS_SIGMA build accumulates into rows 0..L-1 only. */
            memset(S, 0, (size_t) L*H_dim*sizeof(double));
            double itime = omp_get_wtime();
            build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation,
                            N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy);
            if (trace) printf("build sigma took %f seconds to execute \n", omp_get_wtime() - itime);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, H_dim, 1.0, S, H_dim, Q, H_dim, 0.0, G, L);
            build_sigma_s_square(Q, newS, Sdiag, b_array, table, num_links0, n_o_ac, num_alpha, L, N_p, 1.0);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, H_dim, 1.0, newS, H_dim, Q, H_dim, 0.0, Gs, L);
        }

        if (!guess_sigma) {
            /* direct Slater-Condon block build (default) */
            double* Gd  = guess_cmp ? (double*) malloc((size_t)L*L*sizeof(double)) : G;
            double* Gsd = guess_cmp ? (double*) malloc((size_t)L*L*sizeof(double)) : Gs;
            double itime = omp_get_wtime();
            build_guess_blocks_scrules(Gd, Gsd, sel, L, h1e, h2e, d_cmo, Sdiag,
                    N_ac, n_o_ac, n_o_in, N_p, num_alpha, Enuc, dc, omega, d_exp, E_core);
            if (trace) printf("direct guess block build took %f seconds to execute \n", omp_get_wtime() - itime);
            if (guess_cmp) {
                double maxDiag = 0.0, maxOff = 0.0, maxS = 0.0;
                for (int a = 0; a < L; a++)
                    for (int b = 0; b < L; b++) {
                        double gh = fabs(Gd[a*L+b] - G[a*L+b]);
                        double gs = fabs(Gsd[a*L+b] - Gs[a*L+b]);
                        if (a == b) { if (gh > maxDiag) maxDiag = gh; } else { if (gh > maxOff) maxOff = gh; }
                        if (gs > maxS) maxS = gs;
                    }
                printf("[guess-direct] L=%d  Gdiag_max=%.3e  Goff_max=%.3e  Gs_max=%.3e\n", L, maxDiag, maxOff, maxS);
                fflush(stdout);
                cblas_dcopy((MKL_INT)((size_t)L*L), Gd, 1, G, 1);
                cblas_dcopy((MKL_INT)((size_t)L*L), Gsd, 1, Gs, 1);
                free(Gd); free(Gsd);
            }
        }

		/* [guess-cost] instrumentation: which term of a large cold-start
		 * P-space guess actually costs.  Synthetic MKL timings do not
		 * transfer here (the S^2 block spectrum is massively degenerate),
		 * so time the real calls. */
		{ double _t0 = omp_get_wtime();
		symmetric_eigenvalue_problem(Gs, L, theta);
		if (trace) printf("[guess-cost] S2 diag N=%d took %f seconds\n", L, omp_get_wtime() - _t0); }
        //for (int i = 0; i < L; i++) {
        //    printf("%20.12lf\n", theta[i]);
	//}
        //for (int i = 0; i < L; i++) {
        //    for (int j = 0; j < L; j++) {
	//	printf("%20.12lf\n", Gs[i * L + j]);    
	//    }
	//	printf("\n");    
	//}	
        int* spin_idx = (int*) malloc(L*sizeof(int));
        memset(spin_idx, 0, L*sizeof(int));
	int count = 0;
        for (int i = 0; i < L; i++) {
	    if (fabs(theta[i] - target_spin * (target_spin +1)) < 1e-9) {
	       spin_idx[count] = i;
	       count += 1;
            }
            //if (count == nroots) {
	    //   break;	    
	    //}	    
	}
        if (count < nroots) {
            printf(
"\n********************************************************************************\n"
" CI Davidson (spin-adapted) cannot start: not enough spin-pure guess vectors.\n\n"
"   determinants in the guess set               = %d\n"
"   vectors found with S = %.1f  (S(S+1) = %.3f) = %d\n"
"   roots requested                             = %d   <-- need at least this many\n"
"   CI dimension H_dim                          = %zu\n\n"
" WHAT HAPPENED.  S^2 is built and diagonalized over the %d selected\n"
" determinants, and only the eigenvectors whose eigenvalue matches the requested\n"
" S(S+1) to 1e-9 can seed the iteration.  Only %d of the %d came back with the\n"
" requested multiplicity -- the rest belong to other spin states, which this run\n"
" is not solving for.\n\n"
" NOTE: davidson_indim is a PER-ROOT MULTIPLIER.  With nroots = %d,\n"
" davidson_indim = %d selects %d determinants, not %d.\n\n"
" FIX: raise davidson_indim so the guess set spans more determinants, and raise\n"
" davidson_maxdim along with it -- the set is grown to close its spin-coupling\n"
" sets and the grown result must still fit in davidson_maxdim * nroots.\n"
" If this active space genuinely does not hold %d states of this multiplicity,\n"
" request fewer roots or change spin_adaptation instead.\n"
"********************************************************************************\n",
                L, target_spin, target_spin * (target_spin + 1.0), count,
                nroots, H_dim, L, count, L,
                nroots, indim_requested / (nroots ? nroots : 1),
                indim_requested, indim_requested / (nroots ? nroots : 1),
                nroots);
            fflush(stdout);
            exit(1);
	}
	indim = count;
        double* Svec = (double*) malloc(indim *L*sizeof(double));
        memset(Svec, 0, indim *L*sizeof(double));
        for (int i = 0; i < indim ; i++) {
            //printf("%4d\n", spin_idx[i]);
            for (int j = 0; j < L; j++) {
	        Svec[i * L + j] = Gs[j * L + spin_idx[i]];
		//printf("%20.12lf", Svec[i * L + j]);
	    }
		//printf("\n");
	}

        double* Hspin = (double*) malloc(indim *indim *sizeof(double));
        memset(Hspin, 0, indim *indim *sizeof(double));
        double* sigma_spin = (double*) malloc(L*indim *sizeof(double));
        memset(sigma_spin, 0, L*indim *sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, indim , L, 1.0, G, L, Svec, L, 0.0, sigma_spin, indim );
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, indim , indim , L, 1.0, Svec, L, sigma_spin, indim , 0.0, Hspin, indim );

        memset(theta, 0, maxdim*sizeof(double));
	{ double _t0 = omp_get_wtime();   /* [guess-cost] */
	symmetric_eigenvalue_problem(Hspin, indim , theta);
	if (trace) printf("[guess-cost] Hspin diag N=%d took %f seconds\n", indim, omp_get_wtime() - _t0); }
        memset(sigma_spin, 0, L*indim *sizeof(double));
        /* [step7] Algorithm 3 step 7, c_subspace = D_tar,00 b_tar^dagger.
         *
         * LAPACKE_dsyev(LAPACK_ROW_MAJOR,'V',...) returns eigenvectors in the
         * COLUMNS of the row-major array (verified by residual against the
         * pristine matrix), so after the step-6 diagonalization above
         *     Hspin[k*indim + r] = d_r(k),
         * i.e. Hspin IS D_tar,00 with columns = d vectors, matching the
         * algorithm's definition.  Svec[i*L + j] = b_i(j), i.e. Svec = b_tar^dagger.
         * The guess vector wanted for root r is
         *     c_r = sum_k d_r(k) b_k = row r of (D^dagger b^dagger),
         * so the product needs Hspin TRANSPOSED.  The plain D b^dagger written
         * here before (and in the printed algorithm, which omits the dagger on
         * D) puts sum_k d_k(r) b_k in row r instead.
         *
         * WHY THIS WAS INVISIBLE: D is orthogonal, so D b^dagger and
         * D^dagger b^dagger have the SAME ROW SPAN and both have orthonormal
         * rows.  Every one of the `count` vectors is handed to Davidson, whose
         * first subspace diagonalization recovers the same Ritz pairs either
         * way -- identical energies, which is what the A/B below confirms.
         * IT STOPS BEING INVISIBLE the moment only the lowest few roots are
         * kept (a collapsed / hybrid guess): rows 0..nroots-1 of D b^dagger are
         * arbitrary mixtures, not the lowest CSF roots.  Fixing it also makes
         * the iteration-1 subspace Hamiltonian exactly diag(e) (see the
         * [step7] check in the Davidson loop).
         * QED_STEP7_LEGACY=1 restores the untransposed product. */
        int step7_legacy = 0; { const char* e = getenv("QED_STEP7_LEGACY"); if (e && atoi(e) != 0) step7_legacy = 1; }
        cblas_dgemm(CblasRowMajor, step7_legacy ? CblasNoTrans : CblasTrans, CblasNoTrans,
                    indim , L, indim , 1.0, Hspin, indim , Svec, L, 0.0, sigma_spin, L);
        guess_theta_n = indim;
        guess_theta = (double*) malloc((size_t)indim * sizeof(double));
        for (int i = 0; i < indim; i++) guess_theta[i] = theta[i];
        memset(newS, 0, L*H_dim*sizeof(double));
        /* Algorithm 3 step 8, c = P c_subspace: expand the spin-pure P-space
         * roots to full CI length.
         *
         * Q's rows here are still the UNTOUCHED unit-vector guess set -- memset
         * above, then Q[i*H_dim + sel[i]] = 1.0 in the selection loop, and
         * nothing writes Q again until the copy-back below.  So
         *     newS = sigma_spin * Q   =>   newS[r][sel[j]] = sigma_spin[r][j],
         * i.e. the projector P is a SCATTER of indim*L elements, not a dense
         * (indim x L) x (L x H_dim) product.  Measured on c10h8 at indim=4000
         * (L=4014, 1364 spin-pure roots, H_dim=63504) the dgemm was 8.8 s of a
         * 38 s excess; the scatter is O(indim*L).
         * newS was memset to zero immediately above, so untouched entries stay 0.
         * QED_STEP8_DGEMM=1 restores the dense product for A/B. */
        { double _t0 = omp_get_wtime();
        int step8_dgemm = 0; { const char* e = getenv("QED_STEP8_DGEMM"); if (e && atoi(e) != 0) step8_dgemm = 1; }
        if (step8_dgemm) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, indim , H_dim, L, 1.0, sigma_spin, L, Q, H_dim, 0.0, newS, H_dim);
        } else {
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < indim; r++)
                for (int j = 0; j < L; j++)
                    newS[(size_t)r*H_dim + (size_t)sel[j]] = sigma_spin[(size_t)r*L + j];
        }
        if (trace) printf("[guess-cost] step 8 expansion (%s, %d x %d -> %zu) took %f seconds\n",
                          step8_dgemm ? "dgemm" : "scatter", indim, L, H_dim, omp_get_wtime() - _t0); }
	//for (int i = 0; i < indim ; i++) {
	//    printf("%d\n",i);
        //    for (int j = 0; j < H_dim ; j++) {
        //        printf("%20.12lf\n", newS[i*H_dim+j]);	    
        //    }
	//    printf("\n");
        //}
	
        /* Only the rows that are actually part of the subspace need zeroing.
        * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
        * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
        * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
        * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
        * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
        * where L is 181.
        * Here: the loop below overwrites rows 0..indim-1 wholesale from newS; L becomes indim. */
        memset(Q, 0, (size_t) (hybrid ? indim + nroots : indim)*H_dim*sizeof(double));
        if (hybrid) {
            /* HYBRID ASSEMBLY.  Q = [ spin-pure P-space | warm CI vector(s) ].
             *
             * THE ORDER IS P-SPACE FIRST, AND THAT IS A PERFORMANCE DECISION,
             * NOT A QUALITY ONE.  Gram-Schmidt does not change the SPAN, only
             * the basis, and Davidson only sees the span -- so either order
             * gives the same subspace.  What the order does change is sparsity:
             * the P-space vectors are supported entirely inside the L selected
             * determinants, the warm vector is dense over all H_dim, and
             * whichever set is orthogonalized SECOND picks up the other's
             * support.  Warm-first would smear the dense warm vector through
             * all `count` P-space rows and make every one of them dense.
             * That matters because build_sparse_init_sigma (:1883) declines
             * when the union of supports exceeds 8*num_state and the caller
             * then falls back to build_sigma -- one full sigma product PER
             * VECTOR.  Item 219C measured that fallback at 5.34 s vs 0.164 s
             * for 388 guess vectors, per CI solve; across ~89 microiterations
             * that alone would cost more than the entire baseline run.
             * P-space-first keeps `count` rows sparse and makes only the
             * `nroots` warm rows dense.
             *
             * `eigenvecs` still holds the warm vectors at this point: the
             * P-space build above writes Q, G, Gs, newS, sel and friends, and
             * never touches eigenvecs (which is only written at the exits of
             * this routine).  So no separate stash is needed.
             *
             * Hazard 1 of item 218C (spin contamination) is handled by
             * construction: the P-space vectors are S^2 eigenvectors filtered
             * to the target S, the warm vector is a converged CI vector of that
             * same S, and any linear combination of same-S vectors is still an
             * S eigenvector -- so orthogonalization cannot contaminate the set.
             * Hazard 3 (the warm vector is NOT orthogonal to the unit vectors)
             * is what the Gram-Schmidt below exists for. */
            for (int i = 0; i < indim; i++)
                cblas_dcopy((MKL_INT) H_dim, newS + (size_t)i*H_dim, 1, Q + (size_t)i*H_dim, 1);
            for (int i = 0; i < nroots; i++)
                cblas_dcopy((MKL_INT) H_dim, eigenvecs + (size_t)i*H_dim, 1,
                            Q + (size_t)(indim + i)*H_dim, 1);
        } else {
        for (int i = 0; i < indim ; i++) {
	    //double total_spin = check_total_spin(newS +i*H_dim, Sdiag, table, b_array, num_links0, n_o_ac, num_alpha, N_p);
            //printf("%20.12lf\n", total_spin);
            //printf("%20.12lf\n", theta[i]);
            for (int j = 0; j < H_dim ; j++) {
		Q[i * H_dim + j] = newS[i * H_dim + j];
            }
	    //printf("\n");
        }
        }
        //for (int i = 0; i < indim ; i++) {
        //    for (int j = 0; j < indim ; j++) {
        //        double dotval = cblas_ddot(H_dim, Q+i*H_dim, 1, Q+j*H_dim, 1);
	//	printf("%20.12lf\n",dotval);
	//    }
	//    printf("\n");
	//}
	L = indim;
        if (hybrid) {
            /* Orthonormalize the warm vector(s) against the P-space.
             *
             * The P-space rows are ALREADY orthonormal -- they are the rows of
             * D^dagger b^dagger (D orthogonal, b^dagger orthonormal rows; see
             * the [step7] note above), scattered onto the selected determinants
             * by step 8.  So only the trailing nroots warm rows need work, and
             * doing just those keeps this O(nroots * L * H_dim) instead of the
             * O(L^2 * H_dim) a full re-orthogonalization of the whole set would
             * cost.  QED_HYBRID_CHECK=1 verifies the orthonormality assumption
             * rather than trusting it.
             *
             * The pre-normalization norm printed below is a direct measure of
             * HOW MUCH OF THE WARM VECTOR THE P-SPACE DOES NOT ALREADY CONTAIN
             * -- i.e. the redundancy question item 222D raised, read off the
             * geometry instead of inferred from counts. */
            int L_hyb = indim;
            for (int k = 0; k < nroots; k++) {
                double* v = Q + (size_t)(indim + k)*H_dim;
                for (int pass = 0; pass < 2; pass++)   /* MGS + reorthogonalization */
                    for (int i = 0; i < L_hyb; i++) {
                        double dot = cblas_ddot((MKL_INT) H_dim, Q + (size_t)i*H_dim, 1, v, 1);
                        cblas_daxpy((MKL_INT) H_dim, -dot, Q + (size_t)i*H_dim, 1, v, 1);
                    }
                double nrm = sqrt(cblas_ddot((MKL_INT) H_dim, v, 1, v, 1));
                if (trace)
                    printf("[hybrid] warm root %d: component outside the P-space = %.6f\n", k, nrm);
                if (nrm > 1e-6) {
                    cblas_dscal((MKL_INT) H_dim, 1.0/nrm, v, 1);
                    if (L_hyb != indim + k)
                        cblas_dcopy((MKL_INT) H_dim, v, 1, Q + (size_t)L_hyb*H_dim, 1);
                    L_hyb++;
                } else if (trace) {
                    printf("[hybrid] warm root %d DROPPED -- already inside the P-space span\n", k);
                }
            }
            if (hybrid_check) {
                double maxdev = 0.0;
                for (int i = 0; i < L_hyb; i++)
                    for (int j = 0; j <= i; j++) {
                        double d = cblas_ddot((MKL_INT) H_dim, Q + (size_t)i*H_dim, 1, Q + (size_t)j*H_dim, 1);
                        if (i == j) d -= 1.0;
                        if (fabs(d) > maxdev) maxdev = fabs(d);
                    }
                printf("[hybrid] orthonormality of the assembled subspace: max|QQ^T - I| = %.3e\n", maxdev);
                fflush(stdout);
            }
            /* the warm rows are dense; the P-space rows are not.  The sigma
             * build needs to know how many trailing rows to route to the dense
             * path -- see the split at the QED_SPARSE_INIT_SIGMA call. */
            hybrid_dense_tail = L_hyb - indim;
            if (trace)
                printf("[hybrid] P-space %d (requested %d) + warm %d -> subspace %d\n",
                       indim, hybrid_n, nroots, L_hyb);
            L = L_hyb;
            /* The iteration-1 analytic-subspace-H shortcut (item 220A fix 2) is
             * EXACT ONLY for a pure cold guess, where the subspace is precisely
             * the P-space CSF basis and Q(H+penalty)Q^T = diag(e) + shift.  The
             * hybrid mixes in the warm vector and then re-orthogonalizes, so
             * that identity no longer holds.  Drop the snapshot: the guard in
             * the Davidson loop keys off guess_theta being non-NULL. */
            free(guess_theta);
            guess_theta = NULL;
            guess_theta_n = 0;
        }
        free(sel);
        free(G);
        free(Gs);
        free(newS);
        free(spin_idx);
        free(Svec);
        free(Hspin);
        free(sigma_spin);






    }
    else {
	//use input eigenvecs as trial vectors    
        for (int i = 0; i < indim; i++) {
            for (int j = 0; j < H_dim; j++){
		Q[i * H_dim + j] = eigenvecs[i * H_dim + j];    
	    }
	}
    }
    free(Hdiag2);
    int Lmax = maxdim;
    //int rows = num_alpha * (N_ac * (n_o_ac - N_ac) + N_ac + n_o_in);
    //int num_links = rows/num_alpha;
    //maxiter = 5; 
    int L_prev = 0;
    bool restarted = true;
    /* Persistent copy of the subspace Hamiltonian Q(H + penalty)Q^T.  The `G`
     * handed to the eigensolver is DESTROYED by it (it comes back holding the
     * eigenvectors), so the accumulated block has to live somewhere else if
     * later iterations are to reuse it.  Leading dimension is maxdim; only the
     * leading L x L block is ever live.  QED_INCREMENTAL_G=0 disables the reuse
     * and rebuilds the whole block every iteration (the old behaviour). */
    int incremental_g = 1; { const char* e = getenv("QED_INCREMENTAL_G"); if (e && atoi(e) == 0) incremental_g = 0; }
    int guess_g_shortcut = 1; { const char* e = getenv("QED_NO_GUESS_G_SHORTCUT"); if (e && atoi(e) != 0) guess_g_shortcut = 0; }
    int guess_g_compare = 0; { const char* e = getenv("QED_GUESS_G_COMPARE"); if (e && atoi(e) != 0) guess_g_compare = 1; }
    double* Gkeep = (double*) malloc((size_t)maxdim*maxdim*sizeof(double));
    memset(Gkeep, 0, (size_t)maxdim*maxdim*sizeof(double));
    //memset(S, 0, maxdim*H_dim*sizeof(double));
    for (int a = 0; a < maxiter; a++) {
        if (trace) {
            printf("\n");
            printf("ITERATION%4d subspace size%4d\n", a+1, L);
        }

	double itime, ftime, exec_time;
        itime = omp_get_wtime();
        if (!restarted && L_prev > 0 && L > L_prev) {
            // We're continuing - S still has old sigma vectors from last iteration
            // Only compute NEW sigma vectors
            int L_new = L - L_prev;
            
            // Allocate temporary arrays for NEW vectors only (much smaller!)
            double* Q_new = (double*) malloc(L_new * H_dim * sizeof(double));
            double* S_new = (double*) malloc(L_new * H_dim * sizeof(double));
            memset(S_new, 0, L_new * H_dim * sizeof(double));
            
            // Copy new Q vectors to temporary array
            for (int i = 0; i < L_new; i++) {
                cblas_dcopy(H_dim, Q + (L_prev + i) * H_dim, 1, Q_new + i * H_dim, 1);
            }
            
            // Compute sigma for new vectors only
            build_sigma(h1e, h2e, d_cmo, Q_new, S_new, table, table_creation, table_annihilation, 
                        N_ac, n_o_ac, n_o_in, nmo, L_new, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy);
            
            // Apply the active spin treatment for new vectors only
            // (QED_SPIN_METHOD / QED_SPIN_SCALE -- see qed_spin_method above;
            //  no-op when the projection method is selected)
            apply_spin_penalty(Q_new, S_new, L_new, Sdiag, Sdiag_projection,
                               b_array, table, num_links0, n_o_ac, num_alpha, N_p);
            // Copy new sigma vectors to S (starting at position L_prev)
            for (int i = 0; i < L_new; i++) {
                cblas_dcopy(H_dim, S_new + i * H_dim, 1, S + (L_prev + i) * H_dim, 1);
            }
            
            free(Q_new);
            free(S_new);
        
        } else {
            // First iteration or after restart - compute all sigma vectors
            // Only NOW do we clear S
            /* Only the rows that are actually part of the subspace need zeroing.
            * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
            * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
            * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
            * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
            * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
            * where L is 181.
            * Here: build_sigma / the sparse build accumulate into rows 0..L-1 only. */
            memset(S, 0, (size_t) L*H_dim*sizeof(double));
            
            /* QED_SPARSE_INIT_SIGMA: on this branch Q is the guess set -- vectors
             * supported only on the L selected determinants -- so H*Q can be
             * built from sparse columns of H instead of L full sigma products.
             * The spin penalty below is unaffected and still applied. */
            /* The sparse build can carry the FIRST-ORDER penalty itself, so when
             * it succeeds the separate build_sigma_s_square below is SKIPPED --
             * that call is another full pass per vector and was the dominant
             * cost of this branch.  It only knows how to apply scale*S^2, so
             * under penalty2 / projection it is asked for NO penalty (NULL, 0.0)
             * and apply_spin_penalty adds whatever the active method needs. */
            int spin_method_here = qed_spin_method();
            int sparse_does_penalty = (spin_method_here == SPIN_PENALTY1);
            int used_sparse = 0;
            /* HYBRID SPLIT.  The hybrid guess is [sparse P-space | dense warm],
             * and build_sparse_init_sigma judges sparsity over the WHOLE batch
             * it is handed -- one dense row makes the support union ~H_dim, the
             * acceptance rule ncol <= 8*num_state fails, and it declines for
             * ALL L vectors.  That would drop the entire guess onto build_sigma
             * (one full sigma product per vector: 5.34 s vs 0.164 s for 388
             * vectors, item 219C) and make the hybrid unusable at any P-space
             * worth having.  So route the two parts to the two builders.
             * Only the initial guess has this layout; a restart's vectors are
             * all dense and decline anyway. */
            int tail = (a == 0) ? hybrid_dense_tail : 0;
            int Lp = L - tail;
            if (sparse_init && !sparse_cmp) {
                used_sparse = build_sparse_init_sigma(h1e, h2e, d_cmo, Q, S, Lp,
                        N_ac, n_o_ac, n_o_in, N_p, num_alpha, Enuc, dc, omega, d_exp, E_core,
                        sparse_does_penalty ? Sdiag : NULL,
                        sparse_does_penalty ? qed_spin_scale() : 0.0);
                if (used_sparse && tail > 0) {
                    /* the dense warm rows, through the ordinary builder */
                    build_sigma(h1e, h2e, d_cmo, Q + (size_t)Lp*H_dim, S + (size_t)Lp*H_dim,
                                table, table_creation, table_annihilation,
                                N_ac, n_o_ac, n_o_in, nmo, tail, N_p, Enuc, dc, omega,
                                d_exp, E_core, break_degeneracy);
                    /* When the sparse build carried the penalty it did so only
                     * for rows 0..Lp-1, so the tail still needs it.  When it did
                     * not, the `else if (!sparse_does_penalty)` branch below
                     * applies the penalty across all L rows, tail included. */
                    if (sparse_does_penalty)
                        apply_spin_penalty(Q + (size_t)Lp*H_dim, S + (size_t)Lp*H_dim, tail,
                                           Sdiag, Sdiag_projection, b_array, table,
                                           num_links0, n_o_ac, num_alpha, N_p);
                }
            }
            if (!used_sparse) {
                build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation,
                            N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy);
                apply_spin_penalty(Q, S, L, Sdiag, Sdiag_projection,
                                   b_array, table, num_links0, n_o_ac, num_alpha, N_p);
                /* compared against H + scale*S^2, i.e. after the penalty */
                if (sparse_cmp) sparse_init_sigma_compare(h1e, h2e, d_cmo, Q, S, L,
                        N_ac, n_o_ac, n_o_in, N_p, num_alpha, Enuc, dc, omega, d_exp, E_core,
                        sparse_does_penalty ? Sdiag : NULL,
                        sparse_does_penalty ? qed_spin_scale() : 0.0, H_dim);
            } else if (!sparse_does_penalty) {
                /* sparse build gave H*Q only; add penalty2 if that is the method */
                apply_spin_penalty(Q, S, L, Sdiag, Sdiag_projection,
                                   b_array, table, num_links0, n_o_ac, num_alpha, N_p);
            }
        }
        ////////////build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation, 
        ////////////                N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy); 
        //////////////apply  spin penalty 
        ////////////build_sigma_s_square(Q, S, Sdiag, b_array, table, num_links0, n_o_ac, num_alpha, L, N_p, 0.45);
        //////////////build_sigma_s_fourth_power(Q, S, Sdiag_projection, b_array, table, num_links0, n_o_ac, num_alpha, L, N_p, 0.1);
        ftime = omp_get_wtime();
        exec_time = ftime - itime;
        if (trace) printf("build sigma took %f seconds to execute \n", exec_time);
        double* G = (double*) malloc(L*L*sizeof(double));
        memset(G, 0, L*L*sizeof(double));

  
    //for (int a = 0; a < maxdim; a++) {
    //    for (int b = 0; b < H_dim; b++) {
    //    	printf("%d %d %20.12lf\n",a,b, S[a*H_dim +b]);
    //    }
    //    	printf("\n");

    //}
        //print_matrix(S, maxdim, H_dim, 5);
    	//fflush(stdout);


        { double _t0 = omp_get_wtime();
        const char* how = "full";
        /* --- how the subspace Hamiltonian Gkeep(L x L, ld maxdim) is obtained ---
         *
         * (a) COLD-GUESS SHORTCUT, iteration 1 only.  Every guess vector is
         *     supported on the L selected determinants, so Q(H+penalty)Q^T is
         *     the P-space block alone = D^dagger H_tar,00 D + (spin-pure shift) I
         *     = diag(e) + shift, with e the CSF eigenvalues already computed in
         *     Algorithm 3 step 6.  Nothing to compute.  This is only exact
         *     because step 7 is transposed correctly (verified: max|offdiag|
         *     ~1e-13 fixed vs 2.5e-1 legacy) -- see the [step7] note above.
         *     The shift is scale*S(S+1) for penalty1 and 0 for penalty2 /
         *     projection, which do not shift the target spin.
         *
         * (b) INCREMENTAL, whenever the subspace only GREW since last iteration.
         *     Q's and S's first L_prev rows are untouched by gram_schmidt_add
         *     (it writes rows >= L only), so the old block of Gkeep is still
         *     valid and only the L_new new rows need the O(L_new*L*H_dim)
         *     product; the symmetric counterpart columns are copied.  This is
         *     exactly the caching the sigma build above already does.
         *     Measured: 6 x 2.98 s -> ~3 s total at L=1364.
         *
         * (c) FULL rebuild -- first iteration, or after a restart, when the old
         *     rows are gone. */
        if (guess_g_shortcut && guess_theta && a == 0 && L == guess_theta_n) {
            how = "guess-shortcut";
            double shift = qed_spin_pure_shift(target_spin);
            for (int i = 0; i < L; i++) {
                memset(Gkeep + (size_t)i*maxdim, 0, (size_t)L*sizeof(double));
                Gkeep[(size_t)i*maxdim + i] = guess_theta[i] + shift;
            }
            if (guess_g_compare) {
                double* Gchk = (double*) malloc((size_t)L*L*sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, H_dim, 1.0, S, H_dim, Q, H_dim, 0.0, Gchk, L);
                double maxd = 0.0;
                for (int i = 0; i < L; i++)
                    for (int j = 0; j < L; j++) {
                        double d = fabs(Gchk[(size_t)i*L+j] - Gkeep[(size_t)i*maxdim+j]);
                        if (d > maxd) maxd = d;
                    }
                printf("[guess-G] analytic vs computed subspace H: max|diff| = %.3e\n", maxd);
                fflush(stdout);
                free(Gchk);
            }
        } else if (incremental_g && !restarted && L_prev > 0 && L > L_prev) {
            how = "incremental";
            int L_new = L - L_prev;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L_new, L, H_dim, 1.0,
                        S + (size_t)L_prev*H_dim, H_dim, Q, H_dim, 0.0,
                        Gkeep + (size_t)L_prev*maxdim, maxdim);
            for (int i = L_prev; i < L; i++)
                for (int j = 0; j < L_prev; j++)
                    Gkeep[(size_t)j*maxdim + i] = Gkeep[(size_t)i*maxdim + j];
        } else {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, H_dim, 1.0,
                        S, H_dim, Q, H_dim, 0.0, Gkeep, maxdim);
        }
        /* hand the eigensolver its own copy -- it overwrites G with eigenvectors */
        for (int i = 0; i < L; i++)
            cblas_dcopy(L, Gkeep + (size_t)i*maxdim, 1, G + (size_t)i*L, 1);
        double _t1 = omp_get_wtime();
        /* [step7] diagnostic: with the shortcut off this reports how close the
         * computed iteration-1 block is to diag(e) + shift, which is the test
         * that separates a correct step 7 from the legacy one (the energies
         * cannot -- the two guesses span the same space). */
        if (guess_theta && a == 0 && L == guess_theta_n && !(guess_g_shortcut)) {
            double maxoff = 0.0, maxdiag = 0.0;
            double shift = qed_spin_pure_shift(target_spin);
            for (int i = 0; i < L; i++)
                for (int j = 0; j < L; j++) {
                    double v = fabs(G[(size_t)i*L + j]);
                    if (i == j) { double d = fabs(G[(size_t)i*L+i] - (guess_theta[i] + shift));
                                  if (d > maxdiag) maxdiag = d; }
                    else if (v > maxoff) maxoff = v;
                }
            printf("[step7] iteration-1 subspace H: max|offdiag| = %.3e   max|diag - (e + penalty)| = %.3e\n",
                   maxoff, maxdiag);
            fflush(stdout);
        }
        symmetric_eigenvalue_problem(G, L, theta);
        if (trace) printf("[guess-cost] projection L=%d (%s) took %f seconds, subspace diag took %f seconds\n",
                          L, how, _t1 - _t0, omp_get_wtime() - _t1); }
        
        memset(w, 0, nroots*H_dim*sizeof(double));
     
	
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, S, H_dim, 0.0, w, H_dim);
        for (int i = 0; i < nroots; i++) {
            cblas_dgemv(CblasRowMajor, CblasTrans, L, H_dim, -theta[i], Q, H_dim, G+i, L, 1.0, w + i * H_dim, 1);
	}
        memset(unconverged_idx, 0, nroots*sizeof(int));
        memset(convergence_check, 0, nroots*sizeof(bool));
        unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
        getMemory2(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);

        

        if (trace) printf("  ROOT      RESIDUAL NORM        EIGENVALUE          CONVERGENCE\n");
	int unconv = 0;
        for (int i = 0; i < nroots; i++) {
            double dotval = cblas_ddot(H_dim, w+i*H_dim, 1, w+i*H_dim, 1);
            double residual_norm = sqrt(dotval);
	    /* <S^2> is no longer computed here every iteration -- it is a purely
	     * diagnostic quantity (convergence uses residual_norm only) and was
	     * ~12% of CI time. It is now reported once at the solve's exit via
	     * report_final_spin (converged / maxiter blocks below). */
            if (residual_norm < threshold) {
                convergence_check[i] = true;
	    }
	    else {
                unconverged_idx[unconv] = i;
                convergence_check[i] = false;
                unconv += 1;
	    }

            if (trace) printf("%4d %20.12lf %20.12lf          %s\n",i, residual_norm, theta[i], convergence_check[i]?"true":"false");
        }
        
    	fflush(stdout);
	if (unconv == 0) {
            //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);
            cblas_dcopy(nroots, theta, 1, eigenvals, 1);
	    constdouble[4] = 0.0;
	    for (int i = 0; i < nroots; i++) {
                double dum = 0.0;
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];	
		}
	        constdouble[4] += sqrt(dum)/nroots;	
	    }
	    //constdouble[4] = sqrt(dum)/nroots;
            constint[8] = 0;
	    last_ci_iterations = a + 1;
	    last_ci_nroots = nroots;
	    for (int i = 0; i < nroots && i < CI_MAX_ROOTS; i++) {
	        double rr = cblas_ddot(H_dim, w + i * H_dim, 1, w + i * H_dim, 1);
	        last_ci_root_residuals[i] = sqrt(rr);
	    }
	    if (get_ci_print_level() >= 3) printf("converged\n");
	    report_final_spin(eigenvecs, nroots, Sdiag, table, b_array, num_links0, n_o_ac, num_alpha, N_p, target_spin);
    	    fflush(stdout);
	    break;
	}
	if ((a==maxiter-1) && (unconv > 0)) {
	    last_ci_iterations = a + 1;
	    last_ci_nroots = nroots;
	    for (int i = 0; i < nroots && i < CI_MAX_ROOTS; i++) {
	        double rr = cblas_ddot(H_dim, w + i * H_dim, 1, w + i * H_dim, 1);
	        last_ci_root_residuals[i] = sqrt(rr);
	    }
	    printf("Maximum iteration reaches. Please increase maxiter!\n");
            //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);
	    cblas_dcopy(nroots, theta, 1, eigenvals, 1);
	    constdouble[4] = 0.0;
	    for (int i = 0; i < nroots; i++) {
                double dum = 0.0;
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];
		}
	        constdouble[4] += sqrt(dum)/nroots;
	    }
	    //constdouble[4] = sqrt(dum)/nroots;
	    report_final_spin(eigenvecs, nroots, Sdiag, table, b_array, num_links0, n_o_ac, num_alpha, N_p, target_spin);
    	    fflush(stdout);
	    break;
	}
	if (trace && unconv > 0) {
	    printf("unconverged roots\n");
            for (int i = 0; i < unconv; i++) {
	    	printf("%4d", unconverged_idx[i]);
	    }
	    printf("\n");
	}

        /* Third spin treatment: clean the RESIDUAL instead of penalising sigma.
         * Active only under QED_SPIN_METHOD=projection, which leaves sigma
         * untouched -- the three methods are mutually exclusive. */
        if (qed_spin_method() == SPIN_PROJECT) {
            for (int i = 0; i < nroots; i++) {
                first_order_spin_projection(w+i*H_dim, Sdiag, Sdiag_projection, table, b_array, num_links0, n_o_ac, num_alpha,
                    N_p, target_spin);
            }
        }



        //precondition
	double* w2 = (double*) malloc(unconv*H_dim*sizeof(double));
        memset(w2, 0, unconv*H_dim*sizeof(double));

	for (int i = 0; i < unconv; i++) {
            for (int j = 0; j < H_dim; j++) {
                    double dum = theta[unconverged_idx[i]] - Hdiag[j];
		    if (fabs(dum) >1e-14) {
		        w2[i * H_dim + j] = w[unconverged_idx[i] * H_dim + j]/dum;
		    }
		    else {
			w2[i * H_dim + j] = 0.0;
		    }
            }
	    //printf("\n");
        }

        // Update L_prev BEFORE modifying Q
        L_prev = L;

        if (Lmax-L < unconv) {
            if (trace) printf("maximum subspace reaches, restart!\n");
            memset(w, 0, nroots*H_dim*sizeof(double));
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, w, H_dim);
            /* Only the rows that are actually part of the subspace need zeroing.
            * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
            * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
            * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
            * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
            * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
            * where L is 181.
            * Here: restart rewrites exactly nroots+unconv rows below, and L becomes nroots+unconv. */
            memset(Q, 0, (size_t) (nroots + unconv)*H_dim*sizeof(double));

            for (int i = 0; i < nroots; i++) {
                cblas_dcopy(H_dim, w+i*H_dim, 1, Q+i*H_dim, 1);
            }

            for (int i = 0; i < unconv; i++) {
                cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+nroots)*H_dim, 1);
            }
            gram_schmidt_orthogonalization(Q, nroots+unconv, H_dim);
            L = nroots + unconv;
            
            restarted = true;
            L_prev = 0;
        }
        else {
            for (int i = 0; i < unconv; i++) {
                cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+L)*H_dim, 1);
            }
            gram_schmidt_add(Q, L, H_dim, unconv);
            L += unconv;
            
            restarted = false;
            // L_prev stays as the old L value (set above)
        }	




        //////if (Lmax-L < unconv) {
        //////   printf("maximum subspace reaches, restart!\n");		
        //////   memset(w, 0, nroots*H_dim*sizeof(double));
        //////   cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, w, H_dim);
        //////   memset(Q, 0, maxdim*H_dim*sizeof(double));

        //////   for (int i = 0; i < nroots; i++) {
        //////       cblas_dcopy(H_dim, w+i*H_dim, 1, Q+i*H_dim, 1);
	//////   }

        //////   for (int i = 0; i < unconv; i++) {
        //////       cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+nroots)*H_dim, 1);
	//////   }
        //////   gram_schmidt_orthogonalization(Q, nroots+unconv, H_dim);
	//////   L = nroots + unconv; 
	//////}
	//////else {
        //////   for (int i = 0; i < unconv; i++) {
        //////       cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+L)*H_dim, 1);
	//////   }
        //////   gram_schmidt_add(Q, L, H_dim, unconv);
        //////   L += unconv;	   
	//////}




        free(G);
        free(w2);
        if (trace) printf("\n");
    }
    free(Q);
    free(S);
    free(theta);
    free(w);
    free(unconverged_idx);
    free(convergence_check);
    
    free(guess_theta);   /* [step7] NULL on the warm path; free(NULL) is a no-op */
    free(Gkeep);


}






void davidson(double* h1e, double* h2e, double* d_cmo, double* Hdiag, double* eigenvals, double* eigenvecs, int* table, 
		int* table_creation, int* table_annihilation, int *constint, double *constdouble, bool casscf, callback_ build_sigma) {
    //unpack constant

    int N_ac = constint[0]; 
    int n_o_ac = constint[1]; 
    int n_o_in = constint[2]; 
    int nmo = constint[3];    
    int N_p = constint[4];   
    int indim = constint[5];         
    int maxdim = constint[6];
    int nroots = constint[7];
    int maxiter = constint[8];
    
    double Enuc = constdouble[0]; 
    double dc = constdouble[1]; 
    double omega = constdouble[2]; 
    double d_exp = constdouble[3];
    double threshold = constdouble[4];
    double E_core = constdouble[5];
    //printf("fafaaavd %s",casscf?"true":"false");
    const int indim_requested = indim;   /* davidson_indim * nroots, before the casscf override */
    /* QED_SPARSE_INIT_SIGMA: build the first/restart sigma directly from the
     * sparse guess vectors instead of one full sigma product per vector (see
     * build_sparse_init_sigma).  QED_SPARSE_INIT_COMPARE builds it both ways
     * and reports the discrepancy while keeping build_sigma's result. */
    int sparse_init = 0; { const char* e = getenv("QED_SPARSE_INIT_SIGMA");   if (e && atoi(e) != 0) sparse_init = 1; }
    int sparse_cmp  = 0; { const char* e = getenv("QED_SPARSE_INIT_COMPARE"); if (e && atoi(e) != 0) sparse_cmp  = 1; }

    if (casscf == true) {
        //maxiter = 3;
	indim = nroots;
    }
    //printf(" maxiter%d", maxiter);

    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    size_t H_dim = (N_p + 1) * num_alpha * num_alpha;

    /* Same buffer-overflow guard as davidson_spin, minus the spin part.  Q, S
     * and theta below are sized by maxdim but the guess loop writes `indim`
     * vectors into Q, so indim > maxdim is a heap overflow -- it corrupts
     * memory silently and surfaces much later as an all-NaN CI vector.
     * Unlike davidson_spin this routine has no spin-coupling growth loop, so
     * indim cannot creep past what was asked for: reaching here means
     * davidson_indim > davidson_maxdim was set directly.  helper_PFCI.py:3076
     * checks the H_dim bound before calling, but the C backend is also driven
     * directly (cpp_casscf), so enforce both here where the buffers are sized. */
    if ((size_t) indim > H_dim || (size_t) maxdim > H_dim || indim > maxdim) {
        printf(
"\n********************************************************************************\n"
" CI Davidson cannot start: the requested subspace does not fit.\n\n"
"   davidson_indim  * nroots = %d%s\n"
"   davidson_maxdim * nroots = %d   <-- must be >= the initial subspace\n"
"   CI dimension H_dim       = %zu   <-- neither may exceed this\n"
"   nroots = %d\n\n"
" NOTE: davidson_indim and davidson_maxdim are PER-ROOT MULTIPLIERS, so the\n"
" sizes that matter are the products above, not the values you set.\n\n"
" FIX: set davidson_indim and davidson_maxdim to at most %zu (= H_dim / nroots),\n"
" keeping davidson_maxdim > davidson_indim so the subspace has room to expand.\n"
" For a CI space this small, full diagonalization is usually the better choice.\n"
"********************************************************************************\n",
            indim, (casscf && indim != indim_requested) ? " (overridden to nroots for casscf)" : "",
            maxdim, H_dim, nroots,
            nroots ? H_dim / (size_t) nroots : H_dim);
        fflush(stdout);
        exit(1);
    }


    double* Hdiag2 = (double*)malloc(H_dim*sizeof(double));
    cblas_dcopy(H_dim,Hdiag,1,Hdiag2,1);
    double* Q = (double*) malloc(maxdim*H_dim*sizeof(double));
    /* Only the rows that are actually part of the subspace need zeroing.
    * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
    * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
    * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
    * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
    * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
    * where L is 181.
    * Here: L is not in scope yet; the guess fills rows 0..indim-1 (unit vectors write one element per row). */
    memset(Q, 0, (size_t) indim*H_dim*sizeof(double));
    double* S = (double*) malloc(maxdim*H_dim*sizeof(double));

    double* w = (double*) malloc(nroots*H_dim*sizeof(double));
    int* unconverged_idx = (int*) malloc(nroots*sizeof(int));
    bool* convergence_check = (bool*) malloc(nroots*sizeof(bool));
    double* theta = (double*) malloc(maxdim*sizeof(double));
    memset(theta, 0, maxdim*sizeof(double));
        
    double minimum = 0.0;
    int min_pos;
    if (casscf == false) {
        //use unit vectors as guess
        for (int i = 0; i < indim; i++) {
            minimum = Hdiag2[0];
            min_pos = 0;
            for (int j = 1; j < H_dim; j++){
                if (Hdiag2[j] < minimum) {
                    minimum = Hdiag2[j];
                    min_pos = j;
                }
            }
            Q[i * H_dim + min_pos] = 1.0;
            Hdiag2[min_pos] = BIGNUM;
           // lambda_oldp[i] = minimum;
        }
    }
    else {
	//use input eigenvecs as trial vectors    
        for (int i = 0; i < indim; i++) {
            for (int j = 0; j < H_dim; j++){
		Q[i * H_dim + j] = eigenvecs[i * H_dim + j];    
	    }
	}
    }
    free(Hdiag2);
    int L = indim;
    int Lmax = maxdim;
    //int rows = num_alpha * (N_ac * (n_o_ac - N_ac) + N_ac + n_o_in);
    //int num_links = rows/num_alpha;
    //maxiter = 5;

    int L_prev = 0;
    bool restarted = true;
    for (int a = 0; a < maxiter; a++) {
        printf("\n"); 
                
        printf("ITERATION%4d subspace size%4d\n", a+1, L);
        bool break_degeneracy = false;         
        
	double itime, ftime, exec_time;
        itime = omp_get_wtime();
        if (!restarted && L_prev > 0 && L > L_prev) {
            // We're continuing - S still has old sigma vectors from last iteration
            // Only compute NEW sigma vectors
            int L_new = L - L_prev;
            
            // Allocate temporary arrays for NEW vectors only (much smaller!)
            double* Q_new = (double*) malloc(L_new * H_dim * sizeof(double));
            double* S_new = (double*) malloc(L_new * H_dim * sizeof(double));
            memset(S_new, 0, L_new * H_dim * sizeof(double));
            
            // Copy new Q vectors to temporary array
            for (int i = 0; i < L_new; i++) {
                cblas_dcopy(H_dim, Q + (L_prev + i) * H_dim, 1, Q_new + i * H_dim, 1);
            }
            
            // Compute sigma for new vectors only
            build_sigma(h1e, h2e, d_cmo, Q_new, S_new, table, table_creation, table_annihilation, 
                        N_ac, n_o_ac, n_o_in, nmo, L_new, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy);
            
            // Copy new sigma vectors to S (starting at position L_prev)
            for (int i = 0; i < L_new; i++) {
                cblas_dcopy(H_dim, S_new + i * H_dim, 1, S + (L_prev + i) * H_dim, 1);
            }
            
            free(Q_new);
            free(S_new);
        
        } else {
            // First iteration or after restart - compute all sigma vectors
            // Only NOW do we clear S
            /* Only the rows that are actually part of the subspace need zeroing.
            * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
            * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
            * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
            * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
            * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
            * where L is 181.
            * Here: build_sigma / the sparse build accumulate into rows 0..L-1 only. */
            memset(S, 0, (size_t) L*H_dim*sizeof(double));
            
            /* QED_SPARSE_INIT_SIGMA -- see the davidson_spin call site.  Here the
             * guess vectors are plain unit vectors, so each is literally one
             * column of H. */
            int used_sparse = 0;
            if (sparse_init && !sparse_cmp) {
                used_sparse = build_sparse_init_sigma(h1e, h2e, d_cmo, Q, S, L,
                        N_ac, n_o_ac, n_o_in, N_p, num_alpha, Enuc, dc, omega, d_exp, E_core,
                        NULL, 0.0);   /* no spin penalty on the non-spin-adapted path */
            }
            if (!used_sparse) {
                build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation, 
                            N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy);
                if (sparse_cmp) sparse_init_sigma_compare(h1e, h2e, d_cmo, Q, S, L,
                        N_ac, n_o_ac, n_o_in, N_p, num_alpha, Enuc, dc, omega, d_exp, E_core,
                        NULL, 0.0, H_dim);
            }
            
        }
        //build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation, 
        //                N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy); 

        ftime = omp_get_wtime();
        exec_time = ftime - itime;
        printf("build sigma took %f seconds to execute \n", exec_time);
        double* G = (double*) malloc(L*L*sizeof(double));
        memset(G, 0, L*L*sizeof(double));

  
    //for (int a = 0; a < maxdim; a++) {
    //    for (int b = 0; b < H_dim; b++) {
    //    	printf("%d %d %20.12lf\n",a,b, S[a*H_dim +b]);
    //    }
    //    	printf("\n");

    //}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, H_dim, 1.0, S, H_dim, Q, H_dim, 0.0, G, L);
        symmetric_eigenvalue_problem(G, L, theta);
        
        memset(w, 0, nroots*H_dim*sizeof(double));
     
	
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, S, H_dim, 0.0, w, H_dim);
        for (int i = 0; i < nroots; i++) {
            cblas_dgemv(CblasRowMajor, CblasTrans, L, H_dim, -theta[i], Q, H_dim, G+i, L, 1.0, w + i * H_dim, 1);
	}
        memset(unconverged_idx, 0, nroots*sizeof(int));
        memset(convergence_check, 0, nroots*sizeof(bool));
        unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
        getMemory2(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);



        printf("  ROOT      RESIDUAL NORM        EIGENVALUE      CONVERGENCE\n");
	int unconv = 0;
        for (int i = 0; i < nroots; i++) {
            double dotval = cblas_ddot(H_dim, w+i*H_dim, 1, w+i*H_dim, 1);
            double residual_norm = sqrt(dotval);
            if (residual_norm < threshold) {
                convergence_check[i] = true;
	    }
	    else {
                unconverged_idx[unconv] = i;
                convergence_check[i] = false;
                unconv += 1;
	    }

            printf("%4d %20.12lf %20.12lf      %s\n",i, residual_norm, theta[i], convergence_check[i]?"true":"false");
        }
    	fflush(stdout);
	if (unconv == 0) {
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);
            cblas_dcopy(nroots, theta, 1, eigenvals, 1);
	    constdouble[4] = 0.0;
	    for (int i = 0; i < nroots; i++) {
                double dum = 0.0;
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];	
		}
	        constdouble[4] += sqrt(dum)/nroots;
	    }
	    //constdouble[4] = sqrt(dum)/nroots;
            constint[8] = 0;
	    last_ci_iterations = a + 1;
	    printf("converged\n");
    	    fflush(stdout);
	    break;
	}
	if ((a==maxiter-1) && (unconv > 0)) {
	    last_ci_iterations = a + 1;
	    printf("Maximum iteration reaches. Please increase maxiter!\n");
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);
	    cblas_dcopy(nroots, theta, 1, eigenvals, 1);
	    constdouble[4] = 0.0;
	    for (int i = 0; i < nroots; i++) {
                double dum = 0.0;
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];	
		}
		//printf("norm %20.12lf cons %20.12lf\n", sqrt(dum), sqrt(dum)/nroots);
	        constdouble[4] += sqrt(dum)/nroots;
	    }
	    //printf("new residual %20.12lf \n", constdouble[4]);
	    //constdouble[4] = sqrt(dum)/nroots;	
	    break;
	}
	if (unconv > 0) {
	    printf("unconverged roots\n");
            for (int i = 0; i < unconv; i++) {
	    	printf("%4d", unconverged_idx[i]);
	    }
	    printf("\n");
	}
        //precondition
	double* w2 = (double*) malloc(unconv*H_dim*sizeof(double));
        memset(w2, 0, unconv*H_dim*sizeof(double));

	for (int i = 0; i < unconv; i++) {
            for (int j = 0; j < H_dim; j++) {
                double dum = theta[unconverged_idx[i]] - Hdiag[j];
                
                if (fabs(dum) >1e-16) {
		    w2[i * H_dim + j] = w[unconverged_idx[i] * H_dim + j]/dum;
		}
		else {
		    w2[i * H_dim + j] = 0.0;
		}
            }
        }
	// Update L_prev BEFORE modifying Q
        L_prev = L;
        
        if (Lmax-L < unconv) {
            printf("maximum subspace reaches, restart!\n");		
            memset(w, 0, nroots*H_dim*sizeof(double));
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, w, H_dim);
            /* Only the rows that are actually part of the subspace need zeroing.
            * Rows beyond it are always FULLY overwritten by cblas_dcopy before they
            * enter the space (the expansion at 'Q+(i+L)*H_dim' and the restart at
            * 'Q+i*H_dim'/'Q+(i+nroots)*H_dim'), and every consumer dgemm reads only L
            * rows -- so the zeros out there were never read.  Zeroing all maxdim rows
            * first-touches the whole buffer: 600 x 245025 x 8 = 1.18 GiB on CAS(8,12)
            * where L is 181.
            * Here: restart rewrites exactly nroots+unconv rows below, and L becomes nroots+unconv. */
            memset(Q, 0, (size_t) (nroots + unconv)*H_dim*sizeof(double));

            for (int i = 0; i < nroots; i++) {
                cblas_dcopy(H_dim, w+i*H_dim, 1, Q+i*H_dim, 1);
            }

            for (int i = 0; i < unconv; i++) {
                cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+nroots)*H_dim, 1);
            }
            gram_schmidt_orthogonalization(Q, nroots+unconv, H_dim);
            L = nroots + unconv;
            
            restarted = true;
            L_prev = 0;
        }
        else {
            for (int i = 0; i < unconv; i++) {
                cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+L)*H_dim, 1);
            }
            gram_schmidt_add(Q, L, H_dim, unconv);
            L += unconv;
            
            restarted = false;
            // L_prev stays as the old L value (set above)
        }	


        //if (Lmax-L < unconv) {
        //   printf("maximum subspace reaches, restart!\n");		
        //   memset(w, 0, nroots*H_dim*sizeof(double));
        //   cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, w, H_dim);
        //   memset(Q, 0, maxdim*H_dim*sizeof(double));

        //   for (int i = 0; i < nroots; i++) {
        //       cblas_dcopy(H_dim, w+i*H_dim, 1, Q+i*H_dim, 1);
	//   }

        //   for (int i = 0; i < unconv; i++) {
        //       cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+nroots)*H_dim, 1);
	//   }
        //   gram_schmidt_orthogonalization(Q, nroots+unconv, H_dim);
	//   L = nroots + unconv; 
	//}
	//else {
        //   for (int i = 0; i < unconv; i++) {
        //       cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+L)*H_dim, 1);
	//   }
        //   gram_schmidt_add(Q, L, H_dim, unconv);
        //   L += unconv;	   
	//}




        free(G);
        free(w2);
        printf("\n"); 
    }
    free(Q);
    free(S);
    free(theta);
    free(w);
    free(unconverged_idx);
    free(convergence_check);
    


}

/* --- QED_RDM_HOIST ----------------------------------------------------------
   build_active_rdm used to malloc+memset, and then free, two num_dets *
   num_photon * n_o_ac^2 buffers on EVERY call (2 x 282 MB at MgH+ CAS(8,12),
   x ~550 calls per optimization).  icx folds malloc+memset(0) into calloc, so
   the pages are lazy and the ~69k first-touch page faults per buffer are
   charged to whichever loop writes first -- which is why loops A and C (the
   first writers of temp1/temp2) measured 2-3x their B/D counterparts despite
   moving exactly the same 282 MB (see the QED_RDM_PROFILE breakdown).

   These buffers are pure scratch: nothing in them survives the call.  So keep
   them across calls and re-zero instead of re-faulting.  The pool is never
   freed -- it is bounded by the largest CAS the process ever runs, and the
   transient peak was already paying for it.  Serial-region only: this routine
   opens its own `omp parallel for` regions, so it is never itself called from
   inside one.  Set QED_RDM_HOIST=0 to restore the per-call malloc/free. */
static int rdm_hoist_enabled(void) {
    static int v = -1;
    if (v < 0) {                       /* called from the serial region only */
        const char* e = getenv("QED_RDM_HOIST");
        v = (e && *e == '0') ? 0 : 1;                       /* default ON */
    }
    return v;
}
static double* rdm_scratch(int slot, size_t nbytes) {
    static double* buf[2] = {NULL, NULL};
    static size_t  cap[2] = {0, 0};
    if (cap[slot] < nbytes) {
        free(buf[slot]);
        buf[slot] = (double*) malloc(nbytes);
        cap[slot] = buf[slot] ? nbytes : 0;
    }
    return buf[slot];
}

void build_one_rdm(double* z_vector, double* eigvec, double* D, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2, bool z_state) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
    
    int n_occupied = n_o_ac + n_o_in;
    //double* D = (double*) malloc(nmo*nmo*sizeof(double));
    //memset(D, 0, nmo*nmo*sizeof(double));
    
    //inactive block
    for (int i = 0; i < n_o_in; i++) {
	int ii = i * n_occupied + i;
	if (z_state == false) {
            D[ii] = 2.0 * (state_p1 == state_p2);	
	}
	else {
            D[ii] = 0.0;		
	}
    }
    //active block
    for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
        int stride = index_jb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            //print(index_kb,sign1,k,l)
            int pq = (p + n_o_in) * n_occupied + q + n_o_in;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ia * num_alpha + index_jb;
        	    D[pq] += sign * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        int stride = index_ja * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int pq = (p + n_o_in) * n_occupied + q + n_o_in;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ja * num_alpha + index_ib;
        	    D[pq] += sign * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
 
    ////double* D2 = (double*)malloc(nmo*nmo*sizeof(double));
    ////cblas_dcopy(nmo*nmo,D,1,D2,1);
    ////double* theta = (double*)malloc(nmo*sizeof(double));
    ////memset(theta, 0, nmo*sizeof(double));
    ////if (state_p1 == state_p2) {
    ////   symmetric_eigenvalue_problem(D2, nmo, theta);
    ////   for (int i = 0; i < nmo; i++) {
    ////       printf( "%20.12lf\n", theta[i]);    
    ////   }
    ////   printf("%4d%4d%4d\n", state_p1, state_p2, nmo);
    ////}
    ////free(D2);
    ////free(theta);

    ////double dum = 0.0;
    ////for (int p = 0; p < nmo; p++) {
    ////    dum += D[p*nmo+p];
    ////}
    ////printf( "%20.12lf\n", dum);    
    
    //double dum2 = cblas_ddot(nmo*nmo, h1e, 1, D, 1);
    //print trace of 1-rdm or 1-tdm and corresponding trace of H1.D or H1.T
    //printf("%4d -> %4d %20.12lf <%d|H1|%d> = %20.12lf\n", state_p1, state_p2, dum, state_p1, state_p2, dum2);
    //printf("%4d -> %4d %20.12lf", state_p1, state_p2, dum2);
    //return(dum2);
}

void build_two_rdm(double* z_vector, double* eigvec, double* D, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2, bool z_state) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
   
    int n_occupied = n_o_ac + n_o_in;
    /*
    double* D_pq = (double*) malloc(n_occupied*n_occupied*sizeof(double));
    memset(D_pq, 0.0, n_occupied*n_occupied*sizeof(double));
 
    for (int i = 0; i < n_o_in; i++) {
	int ii = i * n_occupied + i;
        D_pq[ii] = 2.0 * (state_p1 == state_p2);	
    }

    for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
        int stride = index_jb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            //print(index_kb,sign1,k,l)
            int pq = (p + n_o_in) * n_occupied + q + n_o_in;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ia * num_alpha + index_jb;
        	    D_pq[pq] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        int stride = index_ja * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int pq = (p + n_o_in) * n_occupied + q + n_o_in;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ja * num_alpha + index_ib;
        	    D_pq[pq] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    for (int p = 0; p < n_o_ac; p++) {
        for (int q = 0; q < n_o_ac; q++) {
            int pq = (p + n_o_in) * n_occupied + q + n_o_in;
	    printf("%20.12lf %d %d\n", D_pq[pq], p, q);
	}
    }

    for (int p = 0; p < n_occupied; p++) {
        for (int q = 0; q < n_occupied; q++) {
            for (int i = 0; i < n_o_in; i++) {
                for (int s = 0; s < n_occupied; s++) {
	            int pq =  p           * n_occupied + q;
	            int is =  i           * n_occupied + s;
                    D[pq * n_occupied * n_occupied + is] = 2.0 * D_pq[p * n_occupied + q] * (i==s) - D_pq[p * n_occupied + s] * (i==q);
                    D[is * n_occupied * n_occupied + pq] = D[pq * n_occupied * n_occupied + is];
	        }
	    }
	}
    }


   free(D_pq); 

    */


    ////////double* D_tu2 = (double*) malloc(n_o_ac*n_o_ac*sizeof(double));
    ////////memset(D_tu2, 0.0, n_o_ac*n_o_ac*sizeof(double));
    ////////for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////    int stride = index_jb * num_links;
    ////////    for (int excitation = 0; excitation < num_links; excitation++) {
    ////////        int index_ib = table[(stride + excitation)*4+0];
    ////////        int sign = table[(stride + excitation)*4+1];
    ////////        int p = table[(stride + excitation)*4+2]; 
    ////////        int q = table[(stride + excitation)*4+3]; 
    ////////        //print(index_kb,sign1,k,l)
    ////////        int pq = p * n_o_ac + q;
    ////////        for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////////            for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                int index_I = index_ia * num_alpha + index_ib;
    ////////                int index_J = index_ia * num_alpha + index_jb;
    ////////    	    D_tu2[pq] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////    			* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////        
    ////////            }
    ////////        }
    ////////    }
    ////////}
    ////////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////    int stride = index_ja * num_links;
    ////////    for (int excitation = 0; excitation < num_links; excitation++) {
    ////////        int index_ia = table[(stride + excitation)*4+0];
    ////////        int sign = table[(stride + excitation)*4+1];
    ////////        int p = table[(stride + excitation)*4+2]; 
    ////////        int q = table[(stride + excitation)*4+3]; 
    ////////        int pq = p * n_o_ac + q;
    ////////        for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////////            for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                int index_I = index_ia * num_alpha + index_ib;
    ////////                int index_J = index_ja * num_alpha + index_ib;
    ////////    	    D_tu2[pq] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////    			* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////        
    ////////            }
    ////////        }
    ////////    }
    ////////}




    /* QED_RDM_HOIST: same process-lifetime scratch pool as
       build_active_rdm -- see the comment above build_one_rdm.  Slots are
       shared across all three RDM builders, which is safe because each is a
       separate serial ctypes call from Python and none nests inside another. */
    size_t scratch_bytes = num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double);
    int hoist = rdm_hoist_enabled();
    double* temp1 = hoist ? rdm_scratch(0, scratch_bytes)
                          : (double*) malloc(scratch_bytes);
    memset(temp1, 0, scratch_bytes);
    double* D_tu = (double*) malloc(n_o_ac*n_o_ac*sizeof(double));
    memset(D_tu, 0.0, n_o_ac*n_o_ac*sizeof(double));
    double* D_tuvw = (double*) malloc(n_o_ac*n_o_ac*n_o_ac*n_o_ac*sizeof(double));
    memset(D_tuvw, 0.0, n_o_ac*n_o_ac*n_o_ac*n_o_ac*sizeof(double));


    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ja = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ka * num_alpha + index_jb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ja * num_alpha + index_kb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, 1.0, 
        	    z_vector + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 0.0, D_tu, n_o_ac*n_o_ac);
    double* temp2 = hoist ? rdm_scratch(1, scratch_bytes)
                          : (double*) malloc(scratch_bytes);
    memset(temp2, 0, scratch_bytes);
    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ka * num_alpha + index_ib;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ia * num_alpha + index_kb;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_o_ac*n_o_ac , n_o_ac*n_o_ac, num_photon*num_dets, 1.0, 
        	    temp2, n_o_ac*n_o_ac,
        	    temp1, n_o_ac*n_o_ac, 0.0, 
    		    D_tuvw, n_o_ac*n_o_ac);
    #pragma omp parallel for
    for (int p = 0; p < n_o_ac; p++) {
        for (int r = 0; r < n_o_ac; r++) {
            for (int q = 0; q < n_o_ac; q++) {
                for (int s = 0; s < n_o_ac; s++) {
	            int pr = (p + n_o_in) * n_occupied + r + n_o_in;    
	            int qs = (q + n_o_in) * n_occupied + s + n_o_in;    
                    D[pr * n_occupied * n_occupied + qs] = D_tuvw[(p*n_o_ac+r) * n_o_ac*n_o_ac + q * n_o_ac + s];    
                }
            }
        }
    }
    for (int q = 0; q < n_o_ac; q++) {
        for (int p = 0; p < n_o_ac; p++) {
            for (int s = 0; s < n_o_ac; s++) {
		int qp = (q + n_o_in) * n_occupied + p + n_o_in;    
		int ps = (p + n_o_in) * n_occupied + s + n_o_in;    
        	D[qp * n_occupied * n_occupied + ps] -= D_tu[q*n_o_ac+s];    
            }
        }
    }
    if (!hoist) {                  /* hoisted buffers live for the process */
        free(temp1);
        free(temp2);
    }
    free(D_tuvw);





















    //inactive block
    //Order 2-rdm D^ki_lj = D_kl,ij
    // D_klij = \sum C_im * C_im (4 * d_ij d_kl - 2 * d_kj d_il) 
    for (int k = 0; k < n_o_in; k++) {
        for (int l = 0; l < n_o_in; l++) {
            for (int i = 0; i < n_o_in; i++) {
                for (int j = 0; j < n_o_in; j++) {
                    int kl = k * n_occupied + l;
                    int ij = i * n_occupied + j;
                    if (z_state == false){
                        D[kl * n_occupied * n_occupied + ij] = (4.0 * (i==j) * (k==l) - 2.0 * (k==j) * (i==l)) * (state_p1 == state_p2);
                    }   
		    else {
                        D[kl * n_occupied * n_occupied + ij] = 0.0;
		    }    
        	}
            }
        }
    }
    //inactive-active blocks
    for (int t = 0; t < n_o_ac; t++) {
        for (int u = 0; u < n_o_ac; u++) {
            for (int i = 0; i < n_o_in; i++) {
	        int tu = (t + n_o_in) * n_occupied + u + n_o_in;
	        int ti = (t + n_o_in) * n_occupied + i;
	        //int it =  i           * n_occupied + t + n_o_in;
	        //int ui = (u + n_o_in) * n_occupied + i;
	        int iu =  i           * n_occupied + u + n_o_in;
	        int ii =  i           * n_occupied + i;
                D[ti * n_occupied * n_occupied + iu] = -D_tu[t * n_o_ac + u];
                D[iu * n_occupied * n_occupied + ti] = -D_tu[t * n_o_ac + u];
                //D[it * n_occupied * n_occupied + iu] = -D_tu[t * n_o_ac + u];
                //D[ti * n_occupied * n_occupied + ui] = -D_tu[t * n_o_ac + u];
                D[tu * n_occupied * n_occupied + ii] = 2.0 * D_tu[t * n_o_ac + u];
                D[ii * n_occupied * n_occupied + tu] = 2.0 * D_tu[t * n_o_ac + u];
                

	    }
	}
    }
   free(D_tu); 
/*
*/
    //////////active-active block
    //////////Eq. 25 in Mol. Phys.,2010, 433–451
    ////////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////    int stride1 = index_ja * num_links;
    ////////    for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
    ////////        int index_ka = table[(stride1 + excitation1)*4+0];
    ////////        int sign1 = table[(stride1 + excitation1)*4+1];
    ////////        int q = table[(stride1 + excitation1)*4+2]; 
    ////////        int s = table[(stride1 + excitation1)*4+3]; 
    ////////        for (int p = 0; p < n_o_ac; p++) {
    ////////            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ka * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_ib;
    ////////    		int qp = (q + n_o_in) * n_occupied + p + n_o_in;
    ////////    		int ps = (p + n_o_in) * n_occupied + s + n_o_in;
    ////////                    D2[qp * n_occupied * n_occupied + ps] += -sign1 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////        int stride2 = index_ka * num_links;
    ////////        for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
    ////////            int index_ia = table[(stride2 + excitation2)*4+0];
    ////////            int sign2 = table[(stride2 + excitation2)*4+1];
    ////////            int p = table[(stride2 + excitation2)*4+2]; 
    ////////            int r = table[(stride2 + excitation2)*4+3]; 
    ////////            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_ib;
    ////////    		int pr = (p + n_o_in) * n_occupied + r + n_o_in;
    ////////    		int qs = (q + n_o_in) * n_occupied + s + n_o_in;
    ////////                    D2[pr * n_occupied * n_occupied + qs] += sign1 * sign2 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}
    ////////for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////    int stride1 = index_jb * num_links;
    ////////    for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
    ////////        int index_kb = table[(stride1 + excitation1)*4+0];
    ////////        int sign1 = table[(stride1 + excitation1)*4+1];
    ////////        int q = table[(stride1 + excitation1)*4+2]; 
    ////////        int s = table[(stride1 + excitation1)*4+3]; 
    ////////        for (int p = 0; p < n_o_ac; p++) {
    ////////            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_kb;
    ////////                    int index_J = index_ia * num_alpha + index_jb;
    ////////    		int qp = (q + n_o_in) * n_occupied + p + n_o_in;
    ////////    		int ps = (p + n_o_in) * n_occupied + s + n_o_in;
    ////////                    D2[qp * n_occupied * n_occupied + ps] += -sign1 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////        int stride2 = index_kb * num_links;
    ////////        for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
    ////////            int index_ib = table[(stride2 + excitation2)*4+0];
    ////////            int sign2 = table[(stride2 + excitation2)*4+1];
    ////////            int p = table[(stride2 + excitation2)*4+2]; 
    ////////            int r = table[(stride2 + excitation2)*4+3]; 
    ////////            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ia * num_alpha + index_jb;
    ////////    		int pr = (p + n_o_in) * n_occupied + r + n_o_in;
    ////////    		int qs = (q + n_o_in) * n_occupied + s + n_o_in;
    ////////                    D2[pr * n_occupied * n_occupied + qs] += sign1 * sign2 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}
    ////////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////    int stride_a = index_ja * num_links;
    ////////    for (int excitation_a = 0; excitation_a < num_links; excitation_a++) {
    ////////        int index_ia = table[(stride_a+ excitation_a)*4+0];
    ////////        int sign_a = table[(stride_a + excitation_a)*4+1];
    ////////        int q = table[(stride_a + excitation_a)*4+2]; 
    ////////        int s = table[(stride_a + excitation_a)*4+3]; 
    ////////        

    ////////        for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////            int stride_b = index_jb * num_links;
    ////////            for (int excitation_b = 0; excitation_b < num_links; excitation_b++) {
    ////////                int index_ib = table[(stride_b + excitation_b)*4+0];
    ////////                int sign_b = table[(stride_b + excitation_b)*4+1];
    ////////                int p = table[(stride_b + excitation_b)*4+2]; 
    ////////                int r = table[(stride_b + excitation_b)*4+3]; 
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_jb;
    ////////     		int pr = (p + n_o_in) * n_occupied + r + n_o_in;
    ////////    		int qs = (q + n_o_in) * n_occupied + s + n_o_in;
    ////////                    D2[pr * n_occupied * n_occupied + qs] += sign_a * sign_b * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////                
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}

    ////////for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////    int stride_b = index_jb * num_links;
    ////////    for (int excitation_b = 0; excitation_b < num_links; excitation_b++) {
    ////////        int index_ib = table[(stride_b+ excitation_b)*4+0];
    ////////        int sign_b = table[(stride_b + excitation_b)*4+1];
    ////////        int q = table[(stride_b + excitation_b)*4+2]; 
    ////////        int s = table[(stride_b + excitation_b)*4+3]; 
    ////////        

    ////////        for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////            int stride_a = index_ja * num_links;
    ////////            for (int excitation_a = 0; excitation_a < num_links; excitation_a++) {
    ////////                int index_ia = table[(stride_a + excitation_a)*4+0];
    ////////                int sign_a = table[(stride_a + excitation_a)*4+1];
    ////////                int p = table[(stride_a + excitation_a)*4+2]; 
    ////////                int r = table[(stride_a + excitation_a)*4+3]; 
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_jb;
    ////////     		int pr = (p + n_o_in) * n_occupied + r + n_o_in;
    ////////    		int qs = (q + n_o_in) * n_occupied + s + n_o_in;
    ////////                    D2[pr * n_occupied * n_occupied + qs] += sign_a * sign_b * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////                
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}
    ////test trace of 2-RDM
    ////double dum = 0.0;
    ////for (int p = 0; p < n_occupied; p++) {
    ////    for (int q = 0; q < n_occupied; q++) {
    ////        int pp = p * n_occupied + p;
    ////        int qq = q * n_occupied + q;
    ////        dum += D[pp*n_occupied*n_occupied+qq];
    ////    }
    ////}
    ////printf( "%20.12lf\n", dum);    
    
}


/* --- QED_RDM_PROFILE --------------------------------------------------------
   Env-gated per-section wall-clock breakdown of build_active_rdm -- the C side
   of `building RDM` in the Python log (helper_PFCI.py:11067,
   build_state_average_rdms, which calls this once per root).  OFF unless
   QED_RDM_PROFILE is set to something other than 0/empty.  Purely diagnostic;
   it changes no arithmetic. */
static int rdm_prof_enabled(void) {
    static int v = -1;
    if (v < 0) {                       /* called from the serial region only */
        const char* e = getenv("QED_RDM_PROFILE");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
}
#define RDMP(label)                                                           \
    do {                                                                      \
        if (rdmp) {                                                           \
            double _r1 = omp_get_wtime();                                     \
            printf("  [RDMP] %-26s %9.4f\n", (label), _r1 - _r0);             \
            fflush(stdout);                                                   \
            _r0 = _r1;                                                        \
        }                                                                     \
    } while (0)

void build_active_rdm(double* eigvec, double* D_tu, double* D_tuvw, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2, double weight) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
    int rdmp = rdm_prof_enabled();
    double _r0 = rdmp ? omp_get_wtime() : 0.0;
   
    size_t scratch_bytes = num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double);
    int hoist = rdm_hoist_enabled();
    double* temp1 = hoist ? rdm_scratch(0, scratch_bytes)
                          : (double*) malloc(scratch_bytes);
    double* temp0 = (double*) malloc(n_o_ac*n_o_ac*sizeof(double));
    memset(temp0, 0.0, n_o_ac*n_o_ac*sizeof(double));
    RDMP("alloc temp1 (282MB)");
    memset(temp1, 0, scratch_bytes);
    RDMP("memset temp1 (282MB)");
   

    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ja = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ka * num_alpha + index_jb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    RDMP("loop A: alpha -> temp1");
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ja * num_alpha + index_kb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    RDMP("loop B: beta -> temp1");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    eigvec + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 0.0, temp0, n_o_ac*n_o_ac);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    eigvec + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 1.0, D_tu, n_o_ac*n_o_ac);
    RDMP("1-RDM dgemms (x2)");
    double* temp2 = hoist ? rdm_scratch(1, scratch_bytes)
                          : (double*) malloc(scratch_bytes);
    RDMP("alloc temp2 (282MB)");
    memset(temp2, 0, scratch_bytes);
    RDMP("memset temp2 (282MB)");
    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ka * num_alpha + index_ib;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    RDMP("loop C: alpha -> temp2");
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ia * num_alpha + index_kb;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    RDMP("loop D: beta -> temp2");
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_o_ac*n_o_ac , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    temp2, n_o_ac*n_o_ac,
        	    temp1, n_o_ac*n_o_ac, 1.0, D_tuvw, n_o_ac*n_o_ac);

    RDMP("2-RDM dgemm (10.2 GFlop)");
    for (int q = 0; q < n_o_ac; q++) {
        for (int p = 0; p < n_o_ac; p++) {
            for (int s = 0; s < n_o_ac; s++) {
        	D_tuvw[(q*n_o_ac+p)*n_o_ac*n_o_ac + p*n_o_ac + s] -= temp0[q*n_o_ac+s];    
            }
        }
    }
    RDMP("trace correction");
    free(temp0);
    if (!hoist) {                      /* hoisted buffers live for the process */
        free(temp1);
        free(temp2);
    }
    RDMP("free (564MB)");
    ////////for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////    int stride = index_jb * num_links;
    ////////    for (int excitation = 0; excitation < num_links; excitation++) {
    ////////        int index_ib = table[(stride + excitation)*4+0];
    ////////        int sign = table[(stride + excitation)*4+1];
    ////////        int p = table[(stride + excitation)*4+2]; 
    ////////        int q = table[(stride + excitation)*4+3]; 
    ////////        int pq = p * n_o_ac + q;
    ////////        for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////////            for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                int index_I = index_ia * num_alpha + index_ib;
    ////////                int index_J = index_ia * num_alpha + index_jb;
    ////////    	    D_tu[pq] += weight * sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////    			* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////        
    ////////            }
    ////////        }
    ////////    }
    ////////}
    ////////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////    int stride = index_ja * num_links;
    ////////    for (int excitation = 0; excitation < num_links; excitation++) {
    ////////        int index_ia = table[(stride + excitation)*4+0];
    ////////        int sign = table[(stride + excitation)*4+1];
    ////////        int p = table[(stride + excitation)*4+2]; 
    ////////        int q = table[(stride + excitation)*4+3]; 
    ////////        int pq = p * n_o_ac + q;
    ////////        for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////////            for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                int index_I = index_ia * num_alpha + index_ib;
    ////////                int index_J = index_ja * num_alpha + index_ib;
    ////////    	    D_tu[pq] += weight * sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////    			* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////        
    ////////            }
    ////////        }
    ////////    }
    ////////}
    ////////
    ////////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////    int stride1 = index_ja * num_links;
    ////////    for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
    ////////        int index_ka = table[(stride1 + excitation1)*4+0];
    ////////        int sign1 = table[(stride1 + excitation1)*4+1];
    ////////        int q = table[(stride1 + excitation1)*4+2]; 
    ////////        int s = table[(stride1 + excitation1)*4+3]; 
    ////////        for (int p = 0; p < n_o_ac; p++) {
    ////////            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ka * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_ib;
    ////////    		int qp = q * n_o_ac + p;
    ////////    		int ps = p * n_o_ac + s;
    ////////                    D_tuvw[qp * n_o_ac * n_o_ac + ps] += -weight * sign1 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////        int stride2 = index_ka * num_links;
    ////////        for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
    ////////            int index_ia = table[(stride2 + excitation2)*4+0];
    ////////            int sign2 = table[(stride2 + excitation2)*4+1];
    ////////            int p = table[(stride2 + excitation2)*4+2]; 
    ////////            int r = table[(stride2 + excitation2)*4+3]; 
    ////////            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_ib;
    ////////    		int pr = p * n_o_ac + r;
    ////////    		int qs = q * n_o_ac + s;
    ////////                    D_tuvw[pr * n_o_ac * n_o_ac + qs] += weight * sign1 * sign2 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}
    ////////for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////    int stride1 = index_jb * num_links;
    ////////    for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
    ////////        int index_kb = table[(stride1 + excitation1)*4+0];
    ////////        int sign1 = table[(stride1 + excitation1)*4+1];
    ////////        int q = table[(stride1 + excitation1)*4+2]; 
    ////////        int s = table[(stride1 + excitation1)*4+3]; 
    ////////        for (int p = 0; p < n_o_ac; p++) {
    ////////            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_kb;
    ////////                    int index_J = index_ia * num_alpha + index_jb;
    ////////    		int qp = q * n_o_ac + p;
    ////////    		int ps = p * n_o_ac + s;
    ////////                    D_tuvw[qp * n_o_ac * n_o_ac + ps] += -weight * sign1 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////        int stride2 = index_kb * num_links;
    ////////        for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
    ////////            int index_ib = table[(stride2 + excitation2)*4+0];
    ////////            int sign2 = table[(stride2 + excitation2)*4+1];
    ////////            int p = table[(stride2 + excitation2)*4+2]; 
    ////////            int r = table[(stride2 + excitation2)*4+3]; 
    ////////            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ia * num_alpha + index_jb;
    ////////    		int pr = p * n_o_ac + r;
    ////////    		int qs = q * n_o_ac + s;
    ////////                    D_tuvw[pr * n_o_ac * n_o_ac + qs] += weight * sign1 * sign2 * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////            
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}
    ////////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////    int stride_a = index_ja * num_links;
    ////////    for (int excitation_a = 0; excitation_a < num_links; excitation_a++) {
    ////////        int index_ia = table[(stride_a+ excitation_a)*4+0];
    ////////        int sign_a = table[(stride_a + excitation_a)*4+1];
    ////////        int q = table[(stride_a + excitation_a)*4+2]; 
    ////////        int s = table[(stride_a + excitation_a)*4+3]; 
    ////////        

    ////////        for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////            int stride_b = index_jb * num_links;
    ////////            for (int excitation_b = 0; excitation_b < num_links; excitation_b++) {
    ////////                int index_ib = table[(stride_b + excitation_b)*4+0];
    ////////                int sign_b = table[(stride_b + excitation_b)*4+1];
    ////////                int p = table[(stride_b + excitation_b)*4+2]; 
    ////////                int r = table[(stride_b + excitation_b)*4+3]; 
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_jb;
    ////////    		int pr = p * n_o_ac + r;
    ////////    		int qs = q * n_o_ac + s;
    ////////                    D_tuvw[pr * n_o_ac * n_o_ac + qs] += weight * sign_a * sign_b * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////                
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}
    ////////for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
    ////////    int stride_b = index_jb * num_links;
    ////////    for (int excitation_b = 0; excitation_b < num_links; excitation_b++) {
    ////////        int index_ib = table[(stride_b+ excitation_b)*4+0];
    ////////        int sign_b = table[(stride_b + excitation_b)*4+1];
    ////////        int q = table[(stride_b + excitation_b)*4+2]; 
    ////////        int s = table[(stride_b + excitation_b)*4+3]; 
    ////////        

    ////////        for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////////            int stride_a = index_ja * num_links;
    ////////            for (int excitation_a = 0; excitation_a < num_links; excitation_a++) {
    ////////                int index_ia = table[(stride_a + excitation_a)*4+0];
    ////////                int sign_a = table[(stride_a + excitation_a)*4+1];
    ////////                int p = table[(stride_a + excitation_a)*4+2]; 
    ////////                int r = table[(stride_a + excitation_a)*4+3]; 
    ////////                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
    ////////                    int index_I = index_ia * num_alpha + index_ib;
    ////////                    int index_J = index_ja * num_alpha + index_jb;
    ////////    		int pr = p * n_o_ac + r;
    ////////    		int qs = q * n_o_ac + s;
    ////////                    D_tuvw[pr * n_o_ac * n_o_ac + qs] += weight * sign_a * sign_b * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
    ////////                		* eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
    ////////                
    ////////                }
    ////////            }
    ////////        }

    ////////    }
    ////////}
    //////Symmetrize 2-rdm
    ////for (int t = 0; t < n_o_ac; t++) {
    ////    for (int u = 0; u < n_o_ac; u++) {
    ////        for (int v = 0; v < n_o_ac; v++) {
    ////            for (int w = 0; w < n_o_ac; w++) {
    ////                int tu = t * n_o_ac + u;
    ////                int ut = u * n_o_ac + t;
    ////                int vw = v * n_o_ac + w;
    ////    	    double dum = 0.5 * (D_tuvw[tu * n_o_ac * n_o_ac + vw] + D_tuvw[ut * n_o_ac * n_o_ac + vw]);
    ////                D_tuvw[tu * n_o_ac * n_o_ac + vw] = dum;
    ////    	}
    ////        }
    ////    }
    ////}
    
}


/* --- build_active_rdm_sa -----------------------------------------------------
   CASSCF-ONLY specialisation of build_active_rdm for the case bra == ket.

   A SEPARATE FUNCTION RATHER THAN A FLAG INSIDE build_active_rdm, deliberately:
   the general routine keeps serving nuclear_grad.py byte-for-byte unchanged, so
   nothing on the analytic-gradient path has to be revalidated when this one is
   tuned.  Only the two CASSCF call sites move over.

   THE IDENTITY THIS EXPLOITS.  In build_active_rdm, loops C/D build temp2 with
   *exactly* the loop nest of loops A/B up to renaming
   (index_ja, index_jb, q, s) -> (index_ia, index_ib, p, r): same `table`, same
   sign, same packing (qs = q*n_o_ac + s against pr = p*n_o_ac + r), same
   traversal order, same OpenMP partition.  The ONLY difference is the source
   slice -- temp1 reads eigvec[state_p2], temp2 reads eigvec[state_p1].  So when
   the bra and ket slices are the same memory, temp2 == temp1 BIT-FOR-BIT, and
   loops C/D, the second 282 MB buffer and its memset are all redundant.  What
   is left, temp1^T * temp1, is SYMMETRIC -- a rank-k update (dsyrk, half the
   flops of the dgemm it replaces).  Set QED_RDM_VERIFY=1 to build temp2 the
   general way and print max|temp1 - temp2|; it must be exactly 0.

   *** THE CONDITION IS ON THE SOURCE SLICES, NOT ON THE STATE INDICES. ***
   nuclear_grad.py:1024-1045 calls the _z variant with state indices (i, i) --
   both 0 -- but TWO DIFFERENT ARRAYS (eigenvec1 / eigenvec2) for a derivative
   coupling.  So `state_p1 == state_p2` does NOT imply bra == ket in general,
   and a guard written that way inside the shared routine would be silently
   wrong there.  Here there is a single eigvec, where the two do coincide, and
   the routine still DEFERS to the general one if the states differ, so it
   cannot be misused into a wrong answer.

   Call sites: helper_PFCI.py:11075 (build_state_average_rdms) and :4575, both
   of which pass (i, i) with one eigvec. */
void build_active_rdm_sa(double* eigvec, double* D_tu, double* D_tuvw, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2, double weight) {
    static int sa_on = -1;
    if (sa_on < 0) {                   /* called from the serial region only */
        const char* e = getenv("QED_RDM_SA");
        sa_on = (e && *e == '0') ? 0 : 1;                    /* default ON */
    }
    if (state_p1 != state_p2 || !sa_on) {   /* not the specialised case -- defer */
        build_active_rdm(eigvec, D_tu, D_tuvw, table, N_ac, n_o_ac, num_photon,
                         state_p1, state_p2, weight);
        return;
    }
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
    int rdmp = rdm_prof_enabled();
    double _r0 = rdmp ? omp_get_wtime() : 0.0;

    size_t scratch_bytes = num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double);
    int hoist = rdm_hoist_enabled();
    double* temp1 = hoist ? rdm_scratch(0, scratch_bytes)
                          : (double*) malloc(scratch_bytes);
    double* temp0 = (double*) malloc(n_o_ac*n_o_ac*sizeof(double));
    memset(temp0, 0.0, n_o_ac*n_o_ac*sizeof(double));
    RDMP("alloc temp1 (282MB)");
    /* --- QED_RDM_FUSE --------------------------------------------------------
       Loops A and B interchanged from (k, excitation, j) to (k, j, excitation).

       WHY.  The write index is index_K = <one loop var>*num_alpha + <the other>,
       and with `excitation` in the middle the j loop walks temp1 at a stride of
       n_o_ac^2 doubles (1152 B at MgH+) writing ONE double per row, then the
       next excitation walks the SAME rows again -- 36 scattered revisits of every
       row.  With `excitation` innermost each row is visited ONCE, its 1152 B sit
       in L1 for all 36 accumulations, and in loop A index_K becomes MONOTONIC so
       temp1's 282 MB is written sequentially.

       Loop A then covers every (photon_p, index_K) exactly once, so it can
       INITIALISE its row instead of accumulating onto a pre-zeroed buffer -- the
       separate 282 MB memset disappears.  Loop B still accumulates on top, as
       before; it is unchanged in that respect.

       BIT-IDENTICAL, not merely equivalent: for a given (index_K, qs) the
       contributing excitations are the same set in the same ascending order in
       both forms, and loop A still runs entirely before loop B.  Verified by
       QED_RDM_FUSE=0/1 legs.  Set QED_RDM_FUSE=0 to restore the original nests
       (and with them the memset). */
    static int fuse_on = -1;
    if (fuse_on < 0) {                 /* called from the serial region only */
        const char* e = getenv("QED_RDM_FUSE");
        fuse_on = (e && *e == '0') ? 0 : 1;                  /* default ON */
    }
    const int nn2 = n_o_ac * n_o_ac;
    if (!fuse_on) memset(temp1, 0, scratch_bytes);
    RDMP("memset temp1 (282MB)");
    if (fuse_on) {
        #pragma omp parallel for
        for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
            int stride = index_ka * num_links;
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
                int index_K = index_ka * num_alpha + index_jb;
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    double* row = temp1
                        + ((size_t)photon_p * num_dets + index_K) * nn2;
                    const double* c = eigvec
                        + ((size_t)state_p2 * num_photon + photon_p) * num_dets;
                    for (int t = 0; t < nn2; t++) row[t] = 0.0;
                    for (int excitation = 0; excitation < num_links; excitation++) {
                        int index_ja = table[(stride + excitation)*4+0];
                        int sign     = table[(stride + excitation)*4+1];
                        int s        = table[(stride + excitation)*4+2];
                        int q        = table[(stride + excitation)*4+3];
                        row[q * n_o_ac + s] +=
                            sign * c[(size_t)index_ja * num_alpha + index_jb];
                    }
                }
            }
        }
    } else {
    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ja = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ka * num_alpha + index_jb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    }
    RDMP("loop A: alpha -> temp1");
    if (fuse_on) {
        #pragma omp parallel for
        for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
            int stride = index_kb * num_links;
            for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
                int index_K = index_ja * num_alpha + index_kb;
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    double* row = temp1
                        + ((size_t)photon_p * num_dets + index_K) * nn2;
                    const double* c = eigvec
                        + ((size_t)state_p2 * num_photon + photon_p) * num_dets;
                    for (int excitation = 0; excitation < num_links; excitation++) {
                        int index_jb = table[(stride + excitation)*4+0];
                        int sign     = table[(stride + excitation)*4+1];
                        int s        = table[(stride + excitation)*4+2];
                        int q        = table[(stride + excitation)*4+3];
                        row[q * n_o_ac + s] +=
                            sign * c[(size_t)index_ja * num_alpha + index_jb];
                    }
                }
            }
        }
    } else {
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ja * num_alpha + index_kb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    }
    RDMP("loop B: beta -> temp1");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    eigvec + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 0.0, temp0, n_o_ac*n_o_ac);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    eigvec + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 1.0, D_tu, n_o_ac*n_o_ac);
    RDMP("1-RDM dgemms (x2)");

    /* QED_RDM_VERIFY -- diagnostic, off by default.  Rebuilds temp2 with the
       general loops C/D and proves the identity above at the REAL problem
       dimensions (standing rule 15: do not settle a kernel question on a
       synthetic shape).  Costs a full second buffer and two extra loops. */
    {
        const char* _v = getenv("QED_RDM_VERIFY");
        if (_v && *_v && *_v != '0') {
            double* temp2 = (double*) malloc(scratch_bytes);
            memset(temp2, 0, scratch_bytes);
    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ka * num_alpha + index_ib;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ia * num_alpha + index_kb;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    
            /* Two candidate relations.  Loops A/B pack qs = table[3]*n + table[2]
               while loops C/D pack pr = table[2]*n + table[3], so the nests write
               the SAME values at TRANSPOSED orbital positions -- the identity to
               expect is temp2[K][x*n+y] == temp1[K][y*n+x], not plain equality.
               Both are measured so the log records which one holds. */
            double maxdiff_id = 0.0, maxdiff_tr = 0.0;
            size_t nrow = num_dets * num_photon;
            for (size_t K = 0; K < nrow; K++) {
                const double* t1 = temp1 + K*n_o_ac*n_o_ac;
                const double* t2 = temp2 + K*n_o_ac*n_o_ac;
                for (int x = 0; x < n_o_ac; x++) {
                    for (int y = 0; y < n_o_ac; y++) {
                        double a = fabs(t2[x*n_o_ac+y] - t1[x*n_o_ac+y]);
                        double b = fabs(t2[x*n_o_ac+y] - t1[y*n_o_ac+x]);
                        if (a > maxdiff_id) maxdiff_id = a;
                        if (b > maxdiff_tr) maxdiff_tr = b;
                    }
                }
            }
            printf("  [RDMV] state %d/%d  max|t2-t1| %.3e   max|t2-t1^T| %.3e\n",
                   state_p1, state_p2, maxdiff_id, maxdiff_tr);
            fflush(stdout);
            free(temp2);
            _r0 = omp_get_wtime();     /* do not charge the check to a section */
        }
    }

    /* THE 2-RDM CONTRACTION, WITHOUT temp2.
       The replaced dgemm was
           D_tuvw[i][j] += weight * sum_K temp2[K][i] * temp1[K][j].
       Loops C/D differ from loops A/B ONLY in the orbital packing -- A/B use
       qs = table[3]*n_o_ac + table[2], C/D use pr = table[2]*n_o_ac + table[3] --
       so with the bra and ket slices identical,
           temp2[K][x*n_o_ac + y] == temp1[K][y*n_o_ac + x]   (verified exactly
       at MgH+ dimensions by QED_RDM_VERIFY; it is NOT plain equality, which is
       what a first reading of the two nests suggests).  Substituting,
           D_tuvw[p*n+r][j] += weight * M[r*n+p][j],  M = temp1^T * temp1.
       M is SYMMETRIC, so it comes from a rank-k update at half the flops of the
       dgemm, and the transpose is absorbed into a row permutation of a 144x144
       block -- free.  temp2, its memset and its 282 MB never exist.
       QED_RDM_SYRK=0 keeps M's build as a dgemm so the two can be timed against
       each other at this same call site (standing rule 15). */
    {
        int n2 = n_o_ac * n_o_ac;
        const char* _s = getenv("QED_RDM_SYRK");
        int use_syrk = !(_s && *_s == '0');
        double* M = (double*) malloc((size_t)n2*n2*sizeof(double));
        if (use_syrk) {
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, n2,
                        num_photon*num_dets, weight, temp1, n2, 0.0, M, n2);
            for (int i = 0; i < n2; i++)          /* mirror upper -> lower */
                for (int j = i+1; j < n2; j++)
                    M[(size_t)j*n2+i] = M[(size_t)i*n2+j];
        } else {
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n2, n2,
                        num_photon*num_dets, weight, temp1, n2, temp1, n2,
                        0.0, M, n2);
        }
        for (int pp = 0; pp < n_o_ac; pp++) {
            for (int rr = 0; rr < n_o_ac; rr++) {
                const double* srcrow = M + (size_t)(rr*n_o_ac + pp)*n2;
                double* dstrow = D_tuvw + (size_t)(pp*n_o_ac + rr)*n2;
                for (int c = 0; c < n2; c++) dstrow[c] += srcrow[c];
            }
        }
        free(M);
    }
    RDMP("2-RDM syrk (5.1 GFlop)");
    for (int q = 0; q < n_o_ac; q++) {
        for (int p = 0; p < n_o_ac; p++) {
            for (int s = 0; s < n_o_ac; s++) {
        	D_tuvw[(q*n_o_ac+p)*n_o_ac*n_o_ac + p*n_o_ac + s] -= temp0[q*n_o_ac+s];    
            }
        }
    }
    RDMP("trace correction");
    free(temp0);
    if (!hoist) {                  /* hoisted buffers live for the process */
        free(temp1);
    }
    RDMP("free (282MB)");
}

void build_active_rdm_z(double* z_vector, double* eigvec, double* D_tu, double* D_tuvw, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2, double weight) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
   
    /* QED_RDM_HOIST: same process-lifetime scratch pool as
       build_active_rdm -- see the comment above build_one_rdm.  Slots are
       shared across all three RDM builders, which is safe because each is a
       separate serial ctypes call from Python and none nests inside another. */
    size_t scratch_bytes = num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double);
    int hoist = rdm_hoist_enabled();
    double* temp1 = hoist ? rdm_scratch(0, scratch_bytes)
                          : (double*) malloc(scratch_bytes);
    memset(temp1, 0, scratch_bytes);
    double* temp0 = (double*) malloc(n_o_ac*n_o_ac*sizeof(double));
    memset(temp0, 0.0, n_o_ac*n_o_ac*sizeof(double));
   

    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ja = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ka * num_alpha + index_jb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int s = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int qs = q * n_o_ac + s;
            for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_K = index_ja * num_alpha + index_kb;
                    int index_J = index_ja * num_alpha + index_jb;
        	    temp1[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + qs] += sign * eigvec[(state_p2 * num_photon + photon_p) * num_dets + index_J];
            
                }
            }
        }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    z_vector + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 0.0, temp0, n_o_ac*n_o_ac);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    z_vector + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 1.0, D_tu, n_o_ac*n_o_ac);
    double* temp2 = hoist ? rdm_scratch(1, scratch_bytes)
                          : (double*) malloc(scratch_bytes);
    memset(temp2, 0, scratch_bytes);
    #pragma omp parallel for
    for (int index_ka = 0; index_ka < num_alpha; index_ka++) {
        int stride = index_ka * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ka * num_alpha + index_ib;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    #pragma omp parallel for
    for (int index_kb = 0; index_kb < num_alpha; index_kb++) {
        int stride = index_kb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int r = table[(stride + excitation)*4+3]; 
            int pr = p * n_o_ac + r;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_K = index_ia * num_alpha + index_kb;
        	    temp2[(photon_p * num_dets + index_K) * n_o_ac * n_o_ac + pr] += sign * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I];
            
                }
            }
        }
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_o_ac*n_o_ac , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    temp2, n_o_ac*n_o_ac,
        	    temp1, n_o_ac*n_o_ac, 1.0, D_tuvw, n_o_ac*n_o_ac);

    for (int q = 0; q < n_o_ac; q++) {
        for (int p = 0; p < n_o_ac; p++) {
            for (int s = 0; s < n_o_ac; s++) {
        	D_tuvw[(q*n_o_ac+p)*n_o_ac*n_o_ac + p*n_o_ac + s] -= temp0[q*n_o_ac+s];    
            }
        }
    }
    free(temp0);
    if (!hoist) {                  /* hoisted buffers live for the process */
        free(temp1);
        free(temp2);
    }
}


void build_photon_electron_one_rdm(double* z_vector, double* eigvec, double* Dpe, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
    int n_occupied = n_o_ac + n_o_in; 
    int N_p = num_photon - 1;


    for (int i = 0; i < n_o_in; i++) {
	int ii = i * n_occupied + i;
        for (int photon_p = 0; photon_p < num_photon; photon_p++) {
            for (int index_I = 0; index_I < num_dets; index_I++) {
                if (N_p == 0) continue;
                if ((0 < photon_p) && (photon_p < N_p)) {
        	    Dpe[ii] += 2.0 * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
		         * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_I];
        	    Dpe[ii] += 2.0 * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
		    	* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_I];
		}
                else if (photon_p == N_p) {
        	    Dpe[ii] += 2.0 * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
		         * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_I];
		}
                else{
        	    Dpe[ii] += 2.0 * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
		    	* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_I];
		}
	    }
	}
    }





    for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
        int stride = index_jb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
	    int pq = (p + n_o_in) * n_occupied + q + n_o_in;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ia * num_alpha + index_jb;
		    if (N_p == 0) continue;
                    if ((0 < photon_p) && (photon_p < N_p)) {
        	        Dpe[pq] += sign * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe[pq] += sign * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe[pq] += sign * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe[pq] += sign * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                }
            }
        }
    }
    for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        int stride = index_ja * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
	    int pq = (p + n_o_in) * n_occupied + q + n_o_in;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ja * num_alpha + index_ib;
                    if (N_p == 0) continue;
                    if ((0 < photon_p) && (photon_p < N_p)) {
        	        Dpe[pq] += sign * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe[pq] += sign * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe[pq] += sign * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe[pq] += sign * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                }
            }
        }
    }
 
}



void build_active_photon_electron_one_rdm(double* eigvec, double* Dpe_tu, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2, double weight) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
   
    int N_p = num_photon - 1;
    for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
        int stride = index_jb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int pq = p * n_o_ac + q;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ia * num_alpha + index_jb;
		    if (N_p == 0) continue;
                    if ((0 < photon_p) && (photon_p < N_p)) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                }
            }
        }
    }
    for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        int stride = index_ja * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int pq = p * n_o_ac + q;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ja * num_alpha + index_ib;
                    if (N_p == 0) continue;
                    if ((0 < photon_p) && (photon_p < N_p)) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                }
            }
        }
    }
 
}



void build_active_photon_electron_one_rdm_z(double* z_vector, double* eigvec, double* Dpe_tu, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2, double weight) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
   
    int N_p = num_photon - 1;
    for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
        int stride = index_jb * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ib = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int pq = p * n_o_ac + q;
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ia * num_alpha + index_jb;
		    if (N_p == 0) continue;
                    if ((0 < photon_p) && (photon_p < N_p)) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                }
            }
        }
    }
    for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        int stride = index_ja * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ia = table[(stride + excitation)*4+0];
            int sign = table[(stride + excitation)*4+1];
            int p = table[(stride + excitation)*4+2]; 
            int q = table[(stride + excitation)*4+3]; 
            int pq = p * n_o_ac + q;
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                for (int photon_p = 0; photon_p < num_photon; photon_p++) {
                    int index_I = index_ia * num_alpha + index_ib;
                    int index_J = index_ja * num_alpha + index_ib;
                    if (N_p == 0) continue;
                    if ((0 < photon_p) && (photon_p < N_p)) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p)   * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe_tu[pq] += sign * weight * sqrt(photon_p+1) * z_vector[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                }
            }
        }
    }
 
}




void build_b_array(int* table1, int* b_array, int num_alpha, int num_links, int n_o_ac) {
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        int stride = index_ib * num_links;
        for (int pq = 0; pq < n_o_ac*n_o_ac; pq++) {
            int q = pq%n_o_ac;
            int p = (pq -q)/n_o_ac;
            for (int excitation = 0; excitation < num_links; excitation++) {
                if ((table1[(stride+excitation)*4+2] == p) && (table1[(stride+excitation)*4+3] == q)) {
                    b_array[index_ib*n_o_ac*n_o_ac*2 + pq*2+0] = table1[(stride+excitation)*4+0];
                    b_array[index_ib*n_o_ac*n_o_ac*2 + pq*2+1] = table1[(stride+excitation)*4+1];
                }
      	    }
        }
    }
}
void build_s_square_matrix(double* S_diag, double* s_square, int* table1, int num_alpha, int num_links) {
    size_t num_dets = num_alpha * num_alpha;
    for (size_t index_J = 0; index_J < num_dets; index_J++) {
        int index_jb = index_J%num_alpha;
        int index_ja = (index_J-index_jb)/num_alpha;
        int stride_b = index_jb * num_links;
         for (int excitation_b = 0; excitation_b < num_links; excitation_b++) {
             int index_ib = table1[(stride_b + excitation_b)*4+0];
             int sign_b = table1[(stride_b + excitation_b)*4+1];
             int q = table1[(stride_b + excitation_b)*4+2]; 
             int p = table1[(stride_b + excitation_b)*4+3]; 
	     int stride_a = index_ja * num_links;

             for (int excitation_a = 0; excitation_a < num_links; excitation_a++) {
                 if ((table1[(stride_a + excitation_a)*4+2] == p) && (table1[(stride_a + excitation_a)*4+3] == q)) {
                     int index_ia = table1[(stride_a + excitation_a)*4+0];
                     int sign_a = table1[(stride_a + excitation_a)*4+1];
	             int index_I = index_ia * num_alpha + index_ib;		 
	             int index_J = index_ja * num_alpha + index_jb;
		     if (index_I != index_J) {
		         s_square[index_I*num_dets +index_J] -= sign_b*sign_a;
		     }
		     else {
		         s_square[index_I*num_dets +index_J] = S_diag[index_I];
		     }
      	         }
      	     }
	}
    }
}
void build_S_diag(double* S_diag, int num_alpha, int nmo, int N_ac,int n_o_ac,int n_o_in, double shift) {
    size_t num_dets = num_alpha * num_alpha;
    int rows = N_ac*(n_o_ac-N_ac+1);
    int cols = 3;
    
    int* Y = (int*) malloc(rows*cols*sizeof(int));
    get_graph(N_ac,n_o_ac,Y);

  
    #pragma omp parallel for
    for (size_t Idet = 0; Idet < num_dets; Idet++) {
        int index_b = Idet%num_alpha;
        int index_a = (Idet-index_b)/num_alpha;
        size_t string_a = index_to_string(index_a,N_ac,n_o_ac,Y);
        size_t string_b = index_to_string(index_b,N_ac,n_o_ac,Y);
    	size_t ab = string_a & string_b;
    	int length;
    	int* double_active = string_to_obtlist(ab, nmo, &length);
    	int dim_d = length+n_o_in;
    	int double_occupation[dim_d];
        for (int i = 0; i < dim_d; i++) {
            if (i < n_o_in) {
    	       double_occupation[i] = i;	    
    	    }
            else {
    	       double_occupation[i] = double_active[i-n_o_in] + n_o_in;	
    	    }
    	}
            
        size_t e = string_a^string_b;
        size_t ea = e&string_a;
        size_t eb = e&string_b;
        int dim_s;
    	int* single_occupation_a = string_to_obtlist(ea, nmo, &dim_s);
    	int* single_occupation_b = string_to_obtlist(eb, nmo, &dim_s);
        for (int i = 0; i < dim_s; i++) {
    	    single_occupation_a[i] += n_o_in;	
    	    single_occupation_b[i] += n_o_in;	
    	}
            
        int* occupation_list_spin = (int*) malloc((dim_d+2*dim_s)*3*sizeof(int));
        memset(occupation_list_spin, 0, (dim_d+2*dim_s)*3*sizeof(int));
        for (int i = 0; i < dim_d; i++) {
            occupation_list_spin[i*3+0] = double_occupation[i];
            occupation_list_spin[i*3+1] = 1; 
            occupation_list_spin[i*3+2] = 1; 
    	}
        for (int i = 0; i < dim_s; i++) {
            occupation_list_spin[(i+dim_d)*3+0] = single_occupation_a[i];
            occupation_list_spin[(i+dim_d)*3+1] = 1; 
            occupation_list_spin[(i+dim_d+dim_s)*3+0] = single_occupation_b[i];
            occupation_list_spin[(i+dim_d+dim_s)*3+2] = 1; 
    	}
    	
        int n_a = 0;
	int n_b =0;
	for (int a = 0; a < (dim_d+2*dim_s); a++) {
            int n_ia = occupation_list_spin[a*3+1]; 
            int n_ib = occupation_list_spin[a*3+2];
	    if (n_ia + n_ib == 1) {
                n_a += n_ia;
	        n_b += n_ib;
	    } 
    	}
	//printf("%4d\n", n_a+n_b);
	//shift is I * <S^2> in equation 4 (J. Chem. Theory Comput. 2017, 13, 4162−4172)
        S_diag[Idet] = 0.25 * (n_a - n_b) * (n_a - n_b) + 0.5 * (n_a+n_b) - shift;
    	free(double_active);
    	free(single_occupation_a);
    	free(single_occupation_b);
    	free(occupation_list_spin);
        //printf("\n");
    }
	free(Y);
}

void build_sigma_s_square_off_diagonal(double* c_vectors, double* c1_vectors, int* b_array, int* table1, int num_alpha, int num_links, int n_o_ac, int photon_p1, int state_p1, 
		int photon_p2, int state_p2, int num_photon, double scale) {
    /*algorithm 2 in J.Chem.TheoryComput.2017, 13, 4162−4172
    //not sure if I implemented it correctly. it's said that b_array contains only the index of j_b, but then we still need a sign of j_b, so I added the sign to array b
    */
    size_t num_dets = num_alpha * num_alpha;
    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
        for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
            int stride_a = index_ia * num_links;
             for (int excitation_a = 0; excitation_a < num_links; excitation_a++) {
                 int index_ja = table1[(stride_a + excitation_a)*4+0];
                 int sign_a = table1[(stride_a + excitation_a)*4+1];
                 int p = table1[(stride_a + excitation_a)*4+2]; 
                 int q = table1[(stride_a + excitation_a)*4+3]; 
                 int qp = q*n_o_ac + p;
		 int index_jb = b_array[index_ib*n_o_ac*n_o_ac*2 + qp*2 + 0];
                 int sign_b = b_array[index_ib*n_o_ac*n_o_ac*2 + qp*2 + 1];
	         int index_I = index_ia * num_alpha + index_ib;		 
	         int index_J = index_ja * num_alpha + index_jb;
                 if ((p !=q) && (index_jb >=0) && (index_I != index_J)) {
	             c1_vectors[(state_p2 * num_photon + photon_p2) * num_dets + index_I] -= sign_a
		      *	scale *	sign_b * c_vectors[(state_p1 * num_photon + photon_p1) * num_dets + index_J]; 	
		 } 
	    }
        }
    }
}

void build_sigma_s_square_diagonal(double* c_vectors, double* c1_vectors, double* S_diag, int num_alpha, int photon_p1, int state_p1,
    int photon_p2, int state_p2, int num_photon, double scale) {
    size_t num_dets = num_alpha * num_alpha;
    for (size_t index_I = 0; index_I < num_dets; index_I++) {
        c1_vectors[(state_p2 * num_photon + photon_p2) * num_dets + index_I] += scale * S_diag[index_I] * c_vectors[(state_p1 * num_photon + photon_p1) * num_dets + index_I];
    }
}

void build_sigma_s_square(double* c_vectors, double *c1_vectors, double* S_diag, int* b_array, int* table1, int num_links, int n_o_ac, int num_alpha, int num_state, int N_p, double scale) {
    int np1 = N_p + 1;
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < num_state; n++) {
        for (int m = 0; m < np1; m++) {
            build_sigma_s_square_off_diagonal(c_vectors, c1_vectors, b_array, table1, num_alpha, num_links, n_o_ac, m, n, m, n, np1, scale);		
            build_sigma_s_square_diagonal(c_vectors, c1_vectors, S_diag, num_alpha, m, n, m, n, np1, scale);		
	}
    }
}

void build_sigma_s_fourth_power(double* c_vectors, double *c1_vectors, double* S_diag, int* b_array, int* table1, int num_links, int n_o_ac, int num_alpha, int num_state, int N_p, double scale) {
    int np1 = N_p + 1;
    size_t num_dets = num_alpha * num_alpha;
    double* sigma_s_square = (double*) malloc(num_dets*sizeof(double));


    //printf("total dim %d", num_alpha*num_alpha*num_state*np1);
    //#pragma omp parallel for collapse(2)
    for (int n = 0; n < num_state; n++) {
        for (int m = 0; m < np1; m++) {
            memset(sigma_s_square, 0, num_dets*sizeof(double));
            build_sigma_s_square_off_diagonal(c_vectors, sigma_s_square, b_array, table1, num_alpha, num_links, n_o_ac, m, n, 0, 0, np1, scale);
            build_sigma_s_square_diagonal(c_vectors, sigma_s_square, S_diag, num_alpha, m, n, 0, 0, np1, scale);

            build_sigma_s_square_off_diagonal(sigma_s_square, c1_vectors, b_array, table1, num_alpha, num_links, n_o_ac, 0, 0, m, n, np1, 1.0);
            build_sigma_s_square_diagonal(sigma_s_square, c1_vectors, S_diag, num_alpha, 0, 0, m, n, np1, 1.0);

        }
    }
    free(sigma_s_square);
}

void gram_schmidt_orthogonalization(double* Q, int rows, int cols) {
    double dotval, normval;
    int k, i, rI;
    int L = 0;
    for (k = 0; k < rows; k++) {
        if (L>0) {
           // Q->print();
            for (i = 0; i < L; i++) {
                dotval = cblas_ddot(cols, &Q[i*cols], 1, &Q[k*cols], 1);
                for (rI = 0; rI < cols; rI++) Q[k * cols + rI] -= dotval * Q[i * cols + rI];
            }
            //reorthogonalization
            for (i = 0; i < L; i++) {
                dotval = cblas_ddot(cols, &Q[i*cols], 1, &Q[k*cols], 1);
                for (rI = 0; rI < cols; rI++) Q[k * cols + rI] -= dotval * Q[i * cols + rI];
            }
        }
        normval = cblas_ddot(cols, &Q[k*cols], 1, &Q[k*cols], 1);
        normval = sqrt(normval);
        //outfile->Printf("trial vector norm%30.18lf\n",normval);
        if (normval > 1e-20) {
            for (rI = 0; rI < cols; rI++) {
                Q[L * cols + rI] = Q[k * cols + rI] / normval;
            }
            L++;
            //outfile->Printf("check orthogonality1\n");
            ////for (int i = 0; i < L; i++) {
            ////    for (int j = 0; j < L; j++) {
            ////        double a = cblas_ddot(cols, &Q[i*cols], 1, &Q[j*cols], 1);
            ////        //outfile->Printf("%d %d %30.16lf",i,j,a);
            ////        if ((i!=j) && (fabs(a)>1e-12)) printf(" detect linear dependency\n");
            ////        //outfile->Printf("\n");
            ////    }
            ////}
        }
    }
}



void gram_schmidt_add(double* Q, int rows, int cols, int rows2) {

    double dotval, normval;
    int k,i, rI;
    for (k = rows; k < rows + rows2; k++) {
        for (i = 0; i < k; i++) {
            dotval = cblas_ddot(cols, &Q[i*cols], 1, &Q[k*cols], 1);
            for (rI = 0; rI < cols; rI++) Q[k * cols + rI] -= dotval * Q[i * cols + rI];
        }
        //reorthogonalization
        for (i = 0; i < k; i++) {
            dotval = cblas_ddot(cols, &Q[i*cols], 1, &Q[k*cols], 1);
            for (rI = 0; rI < cols; rI++) Q[k * cols + rI] -= dotval * Q[i * cols + rI];
        }
        normval = cblas_ddot(cols, &Q[k*cols], 1, &Q[k*cols], 1);
        normval = sqrt(normval);
        //outfile->Printf("trial vector norm2%30.18lf\n",normval);
        if (normval > 1e-20) {
            //printf("normval%20.12lf\n",normval);
            for (rI = 0; rI < cols; rI++) {
                Q[k * cols + rI] = Q[k * cols + rI] / normval;
            }
            //outfile->Printf("check orthogonality2\n");
            ////for (int i = 0; i < L; i++) {
            ////    for (int j = 0; j < L; j++) {
            ////        double a = cblas_ddot(N, Q[i], 1, Q[j], 1);
            ////        //outfile->Printf("%d %d %30.16lf",i,j,a);
            ////        if (i!=j && fabs(a)>1e-12) outfile->Printf(" detect linear dependency\n");
            ////        //outfile->Printf("\n");
            ////    }
            ////}
        }
    }

}
void getMemory2(
   unsigned long* currRealMem, unsigned long*  peakRealMem,
   unsigned long* currVirtMem, unsigned long*  peakVirtMem) {

    // stores each word in status file
    char buffer[1024] = "";

    // linux file contains this-process info
    FILE* file = fopen("/proc/self/status", "r");

    // read the entire file
    while (fscanf(file, " %1023s", buffer) == 1) {

        if (strcmp(buffer, "VmRSS:") == 0) {
            fscanf(file, " %ld", currRealMem);
        }
        if (strcmp(buffer, "VmHWM:") == 0) {
            fscanf(file, " %ld", peakRealMem);
        }
        if (strcmp(buffer, "VmSize:") == 0) {
            fscanf(file, " %ld", currVirtMem);
        }
        if (strcmp(buffer, "VmPeak:") == 0) {
            fscanf(file, " %ld", peakVirtMem);
        }
    }
    fclose(file);
    /* Per-iteration memory readout: diagnostic only (callers still get the
     * values through the out-pointers above), so print it only at the highest
     * level. */
    if (get_ci_print_level() >= 3) {
        printf("resident%20.12lf peak resident%20.12lf\n",(double)*currRealMem/1024.0/1024.0,(double)*peakRealMem/1024.0/1024.0);
    }
}

