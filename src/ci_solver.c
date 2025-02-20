
#include <stdlib.h>
#include <stdio.h>
#include "ci_solver.h"
#include <string.h>
#include <math.h>
#include<omp.h>
#include<time.h>
#include <Accelerate/Accelerate.h>
#include <unistd.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BIGNUM 1E100

// Add transpose function since accelerate will expect column major ordering
void transpose_matrix(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double temp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = temp;
        }
    }
}

void matrix_product(double* A, double* B, double* C, int m, int n, int k) {
     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
}

int binomialCoeff(int n, int k) {
    int C[k + 1];
    memset(C, 0, sizeof(C));
    C[0] = 1; // nC0 is 1
    for (int i = 1; i <= n; i++) {
        for (int j = MIN(i, k); j > 0; j--)
            C[j] = C[j] + C[j - 1];
    }
    return C[k];
}

// Other functions remain unchanged, ensuring they use OpenBLAS-compatible calls.

void build_H_diag(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha, int nmo, int n_act_a, int n_act_orb, int n_in_a, double omega, double Enuc, double dc, int* Y) {
    size_t num_dets = num_alpha * num_alpha;
    int np1 = N_p + 1;
    #pragma omp parallel for num_threads(16)
    for (size_t index_photon_det = 0; index_photon_det < np1 * num_dets; index_photon_det++) {
        size_t Idet = index_photon_det % num_dets;
        int m = (index_photon_det - Idet) / num_dets;
        int start = m * num_dets;
    
        int index_b = Idet % num_alpha;
        int index_a = (Idet - index_b) / num_alpha;
        size_t string_a = index_to_string(index_a, n_act_a, n_act_orb, Y);
        size_t string_b = index_to_string(index_b, n_act_a, n_act_orb, Y);
        size_t ab = string_a & string_b;
        int length;
        int* double_active = string_to_obtlist(ab, nmo, &length);
        int dim_d = length + n_in_a;
        int double_occupation[dim_d];
        for (int i = 0; i < dim_d; i++) {
            if (i < n_in_a) {
                double_occupation[i] = i;    
            } else {
                double_occupation[i] = double_active[i - n_in_a] + n_in_a;    
            }
        }

        size_t e = string_a ^ string_b;
        size_t ea = e & string_a;
        size_t eb = e & string_b;
        int dim_s;
        int* single_occupation_a = string_to_obtlist(ea, nmo, &dim_s);
        int* single_occupation_b = string_to_obtlist(eb, nmo, &dim_s);
        for (int i = 0; i < dim_s; i++) {
            single_occupation_a[i] += n_in_a;
            single_occupation_b[i] += n_in_a;
        }
        
        double c = 0;
        for (int a = 0; a < (dim_d + 2 * dim_s); a++) {
            int i = double_occupation[a];
            int n_i = 1; 
            c += n_i * h1e[i * nmo + i];
        }
        H_diag[Idet + start] = c + m * omega + Enuc + dc;
    }
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




void build_H_diag_cas(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha,int nmo, int n_act_a,int n_act_orb,int n_in_a, double E_core, double omega, double Enuc, double dc, int* Y) {
    size_t num_dets = num_alpha * num_alpha;
    int np1 = N_p + 1;
    int n_occupied = n_act_orb + n_in_a;
    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
    for (size_t index_photon_det = 0; index_photon_det < np1*num_dets; index_photon_det++) {
    	//printf("%d\n", index_photon_det);
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
    printf("get_graph took %f seconds to execute \n", time_taken);

    rows = num_alpha*(N*(n_o-N)+N+n_in_a);
    cols = 4;
    double itime, ftime, exec_time;
    itime = omp_get_wtime();
    //build_H_diag(h1e, h2e, H_diag, N_p, num_alpha, nmo, N, n_o, n_in_a, omega, Enuc, dc,Y);   
    build_H_diag_cas_spin(h1e, h2e, H_diag, N_p, num_alpha, nmo, N, n_o, n_in_a, E_core, omega, Enuc, dc,Y, target_spin);   
    ftime = omp_get_wtime();
    time_taken = ((double)t)/CLOCKS_PER_SEC;
    exec_time = ftime - itime;
    printf("buil_H_diag took %f seconds to execute \n", exec_time);

    t = clock();
    single_replacement_list(num_alpha, N, n_o, Y, table);   
    single_creation_list2(N,n_o, n_in_a, Y,table_creation);
    single_annihilation_list2(N,n_o, n_in_a, Y,table_annihilation);
    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("single_replacement_list took %f seconds to execute \n", time_taken);

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
    #pragma omp parallel for num_threads(12) collapse(2)
    for (int n = 0; n < num_state; n++) {
        for (int m = 0; m < np1; m++) {
            sigma12(h1e, h2e, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, n_o_ac, n_o_in, m, n, np1); 
            sigma3(h2e, c_vectors, c1_vectors, table, table_creation, table_annihilation, 
       	   N_ac, n_o_ac, n_o_in, nmo, m, n, np1);
            double someconstant = m * omega + Enuc + dc + E_core;
            if (break_degeneracy == true) {
               someconstant = m * (omega + 1) + Enuc + dc + E_core;
            }
            constant_terms_contraction(c_vectors, c1_vectors, num_alpha, someconstant, m, m, n, np1);    
	    if (N_p == 0) continue;
            if ((0 < m) && (m < N_p)) {
                someconstant = -sqrt(m * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m-1, m, n, np1);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);    
                someconstant = -sqrt((m+1) * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m+1, m, n, np1);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);    
            }
            else if (m == N_p) {
                someconstant = -sqrt(m * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m-1, m, n, np1);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);   
            }
            else {
                someconstant = -sqrt((m+1) * omega/2);
                sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, n_o_ac, n_o_in, someconstant, m+1, m, n, np1);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);   
            }
        }
    }
}
       


void sigma3(double* h2e, double* c_vectors, double* c1_vectors, int* table,  int* table_creation, int* table_annihilation, 
		 int N_ac, int n_o_ac, int n_o_in, int nmo, int photon_p, int state_p, int num_photon) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_alpha1 = binomialCoeff(n_o_ac, N_ac-1);
    size_t num_dets = num_alpha * num_alpha;
    
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    int num_links1 = n_o_ac-N_ac+1;
    int num_links2 = N_ac;
    int n_occupied = n_o_ac + n_o_in;
    double* D = (double*) malloc(num_alpha1 * n_o_ac * num_alpha*sizeof(double));
    memset(D, 0, num_alpha1 * n_o_ac * num_alpha * sizeof(double));
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


    //printf("\n");
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
		int photon_p, int state_p, int num_photon) {
    size_t num_dets = num_alpha * num_alpha;
    double* F = (double*) malloc(num_alpha*sizeof(double));

    double* s_resize = (double*) malloc(num_alpha*sizeof(double));
    int n_occupied = n_o_ac + n_o_in;

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
        //for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
        //    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
        //        int index_I = index_ia * num_alpha + index_ib;
        //        int index_J = index_ia * num_alpha + index_jb;
        //        c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += F[index_jb] * c_vectors[
        //       	 (state_p * num_photon + photon_p) * num_dets + index_J];
        //    }
        //}

    }


    double* c_resize = (double*) malloc(num_dets*sizeof(double));
    
    for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
            int index_J = index_ja * num_alpha + index_ib;
            int index_Jt = index_ib * num_alpha + index_ja;
            c_resize[index_Jt] = c_vectors[
       	 (state_p * num_photon + photon_p) * num_dets + index_J];
        }
    }
    
    
    
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


        //for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
        //    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        //        int index_I = index_ia * num_alpha + index_ib;
        //        int index_J = index_ja * num_alpha + index_ib;
        //        c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += F[index_ja] * c_vectors[
        //       	 (state_p * num_photon + photon_p) * num_dets + index_J];
        //    }
        //}

    }
    free(F); 
    free(c_resize); 
    free(s_resize);
    
}
void sigma_dipole(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int n_o_ac, int n_o_in, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon) {

     size_t num_dets = num_alpha * num_alpha;
     int n_occupied = n_o_ac + n_o_in;
     double* F = (double*) malloc(num_alpha*sizeof(double));

     double* s_resize = (double*) malloc(num_alpha*sizeof(double));

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

         //for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
         //    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
         //        int index_I = index_ia * num_alpha + index_ib;
         //        int index_J = index_ia * num_alpha + index_jb;
         //        c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += someconstant * F[index_jb] * c_vectors[
	 //       	 (state_p * num_photon + photon_p1) * num_dets + index_J];

	 //    }
	 //}

     }
     
     double* c_resize = (double*) malloc(num_dets*sizeof(double));
     
     for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
         for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
             int index_J = index_ja * num_alpha + index_ib;
             int index_Jt = index_ib * num_alpha + index_ja;
             c_resize[index_Jt] = c_vectors[
        	 (state_p * num_photon + photon_p1) * num_dets + index_J];
         }
     }

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

         //for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
         //    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
         //        int index_I = index_ia * num_alpha + index_ib;
         //        int index_J = index_ja * num_alpha + index_ib;
         //        c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += someconstant * F[index_ja] * c_vectors[
	 //       	 (state_p * num_photon + photon_p1) * num_dets + index_J];

	 //    }
	 //}

     }

     free(F); 
     free(c_resize); 
     free(s_resize); 

}
void constant_terms_contraction(double* c_vectors,double* c1_vectors,int num_alpha, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon) {
     size_t num_dets = num_alpha * num_alpha;
     for (size_t index_I = 0; index_I < num_dets; index_I++) {
         c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += someconstant * c_vectors[(state_p * num_photon + photon_p1) * num_dets + index_I];
     }
}

void symmetric_eigenvalue_problem(double* A, int N, double* eig) {

    double work_query;
    int lwork = -1;
    int n = N, lda = N, info;

    // Transpose the input matrix (row-major to column-major)
    transpose_matrix(A, n);
    // Perform workspace query
    char jobz = 'V'; // Compute eigenvalues and eigenvectors
    char uplo = 'U'; // Upper triangle of A is stored
    dsyev_(&jobz, &uplo, &n, A, &lda, eig, &work_query, &lwork, &info);

    // Allocate workspace
    lwork = (int)work_query;
    double *work = (double *)malloc(lwork * sizeof(double));

    // Solve eigenproblem
    dsyev_(&jobz, &uplo, &n, A, &lda, eig, work, &lwork, &info);
    // Check for convergence
    if (info > 0) {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

    // Transpose the eigenvectors (column-major to row-major)
    transpose_matrix(A, n);
    // Free workspace
    free(work);


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
    printf("Complete Davidson in %f seconds\n", exec_time);
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
    printf("target_spin %20.12lf\n",target_spin); 
    printf(" indim%d", indim); 
    if (casscf == true) {
        //maxiter = 3;
        indim = nroots;
    }
    else {
    //getting all configurations in the spin coupling set
        while((Hdiag[index_Hdiag[indim]] - Hdiag[index_Hdiag[indim-1]] < 1e-10) && (indim < H_dim)) {
            indim += 1;
        }
    }
    printf(" maxiter%d", maxiter); 
    

    int num_links0 = N_ac*(n_o_ac-N_ac)+N_ac;
    //for (int i = 0; i < H_dim; i++) {
    //    //printf("%4d\n", index_Hdiag[i]);    
    //    printf("%4d %20.12lf\n", index_Hdiag[i], Hdiag[index_Hdiag[i]]);    
    //}
    
    int L = indim;
    printf(" indim%d H_dim %zu\n", indim, H_dim); 
    double* Hdiag2 = (double*)malloc(H_dim*sizeof(double));
    cblas_dcopy(H_dim,Hdiag,1,Hdiag2,1);
    double* Q = (double*) malloc(maxdim*H_dim*sizeof(double));
    memset(Q, 0, maxdim*H_dim*sizeof(double));
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

        memset(S, 0, maxdim*H_dim*sizeof(double));
        
	double itime, ftime, exec_time;
        itime = omp_get_wtime();

        build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation, 
                        N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy); 

        ftime = omp_get_wtime();
        exec_time = ftime - itime;
        printf("build sigma took %f seconds to execute \n", exec_time);
        double* G = (double*) malloc(L*L*sizeof(double));
        memset(G, 0, L*L*sizeof(double));
        double* newS = (double*) malloc(L*H_dim*sizeof(double));
        memset(newS, 0, L*H_dim*sizeof(double));
        double* Gs = (double*) malloc(L*L*sizeof(double));
        memset(Gs, 0, L*L*sizeof(double));

  
    //for (int a = 0; a < maxdim; a++) {
    //    for (int b = 0; b < H_dim; b++) {
    //    	printf("%d %d %20.12lf\n",a,b, S[a*H_dim +b]);
    //    }
    //    	printf("\n");

    //}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, H_dim, 1.0, S, H_dim, Q, H_dim, 0.0, G, L);
 
        
        build_sigma_s_square(Q, newS, Sdiag, b_array, table, num_links0, n_o_ac, num_alpha, L, N_p, 1.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, H_dim, 1.0, newS, H_dim, Q, H_dim, 0.0, Gs, L);
	
	symmetric_eigenvalue_problem(Gs, L, theta);
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
           printf("need bigger initial dim to get enough roots");
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
	symmetric_eigenvalue_problem(Hspin, indim , theta);
        memset(sigma_spin, 0, L*indim *sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, indim , L, indim , 1.0, Hspin, indim , Svec, L, 0.0, sigma_spin, L);
        memset(newS, 0, L*H_dim*sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, indim , H_dim, L, 1.0, sigma_spin, L, Q, H_dim, 0.0, newS, H_dim);
	//for (int i = 0; i < indim ; i++) {
	//    printf("%d\n",i);
        //    for (int j = 0; j < H_dim ; j++) {
        //        printf("%20.12lf\n", newS[i*H_dim+j]);	    
        //    }
	//    printf("\n");
        //}
	
        memset(Q, 0, maxdim*H_dim*sizeof(double));
        for (int i = 0; i < indim ; i++) {
	    //double total_spin = check_total_spin(newS +i*H_dim, Sdiag, table, b_array, num_links0, n_o_ac, num_alpha, N_p);
            //printf("%20.12lf\n", total_spin);	    
            //printf("%20.12lf\n", theta[i]);	    
            for (int j = 0; j < H_dim ; j++) {
		Q[i * H_dim + j] = newS[i * H_dim + j];    
            }
	    //printf("\n");
        }
        //for (int i = 0; i < indim ; i++) {
        //    for (int j = 0; j < indim ; j++) {
        //        double dotval = cblas_ddot(H_dim, Q+i*H_dim, 1, Q+j*H_dim, 1);
	//	printf("%20.12lf\n",dotval);
	//    }
	//    printf("\n");
	//}
	L = indim;
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
    for (int a = 0; a < maxiter; a++) {
        printf("\n"); 
                
        printf("ITERATION%4d subspace size%4d\n", a+1, L);
        memset(S, 0, maxdim*H_dim*sizeof(double));
        
	double itime, ftime, exec_time;
        itime = omp_get_wtime();

        build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation, 
                        N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy); 
        //apply  spin penalty 
        build_sigma_s_square(Q, S, Sdiag, b_array, table, num_links0, n_o_ac, num_alpha, L, N_p, 0.45);
        //build_sigma_s_fourth_power(Q, S, Sdiag_projection, b_array, table, num_links0, n_o_ac, num_alpha, L, N_p, 0.1);
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
        //getMemory2(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);

        

        printf("  ROOT      RESIDUAL NORM        EIGENVALUE          WFN TOTAL SPIN      CONVERGENCE\n");
	int unconv = 0;
        for (int i = 0; i < nroots; i++) {
            double dotval = cblas_ddot(H_dim, w+i*H_dim, 1, w+i*H_dim, 1);
            double residual_norm = sqrt(dotval);
	    double total_spin = check_total_spin(eigenvecs +i*H_dim, Sdiag, table, b_array, num_links0, n_o_ac, num_alpha, N_p);
            if (residual_norm < threshold) {
                convergence_check[i] = true;
	    }
	    else {
                unconverged_idx[unconv] = i;
                convergence_check[i] = false;
                unconv += 1;
	    }

            printf("%4d %20.12lf %20.12lf %20.12lf          %s\n",i, residual_norm, theta[i], total_spin, convergence_check[i]?"true":"false");
        }
        
    	fflush(stdout);
	if (unconv == 0) {
            //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);
            cblas_dcopy(nroots, theta, 1, eigenvals, 1);
            double dum = 0.0;
	    for (int i = 0; i < nroots; i++) {
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];	
		}
	    }
	    constdouble[4] = sqrt(dum);	
	    printf("converged\n");
    	    fflush(stdout);
	    break;
	}
	if ((a==maxiter-1) && (unconv > 0)) {
	    printf("Maximum iteration reaches. Please increase maxiter!\n");
            //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);
	    cblas_dcopy(nroots, theta, 1, eigenvals, 1);
            double dum = 0.0;
	    for (int i = 0; i < nroots; i++) {
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];	
		}
	    }
	    constdouble[4] = sqrt(dum);	
	    break;
	}
	if (unconv > 0) {
	    printf("unconverged roots\n");
            for (int i = 0; i < unconv; i++) {
	    	printf("%4d", unconverged_idx[i]);
	    }
	    printf("\n");
	}

        //for (int i = 0; i < nroots; i++) {
	//    first_order_spin_projection(w+i*H_dim, Sdiag, Sdiag_projection, table, b_array, num_links0, n_o_ac, num_alpha,
        //       N_p, target_spin);
	//}



        //precondition
	double* w2 = (double*) malloc(unconv*H_dim*sizeof(double));
        memset(w2, 0, unconv*H_dim*sizeof(double));

	for (int i = 0; i < unconv; i++) {
            for (int j = 0; j < H_dim; j++) {
                //for (int k = 0; k < L; k++) {
                    double dum = theta[unconverged_idx[i]] - Hdiag[j];
		    //if (i ==2) {
		    ////printf("%20.12lf %20.12lf %20.12lf\n", dum, theta[unconverged_idx[i]], Hdiag[j]);
		    //printf("%20.16lf %20.16lf %20.16lf\n", dum, w[unconverged_idx[i] * H_dim + j], w[unconverged_idx[i] * H_dim + j]/dum);
		    //}
                    if (fabs(dum) >1e-14) {
                    //if (fabs(dum) >1e-16) {
		        w2[i * H_dim + j] = w[unconverged_idx[i] * H_dim + j]/dum;
		    }
		    else {
			w2[i * H_dim + j] = 0.0;
		    }
                //}
            }
	    //printf("\n");
        }
	
		
        if (Lmax-L < unconv) {
           printf("maximum subspace reaches, restart!\n");		
           memset(w, 0, nroots*H_dim*sizeof(double));
           cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, w, H_dim);
           memset(Q, 0, maxdim*H_dim*sizeof(double));

           for (int i = 0; i < nroots; i++) {
               cblas_dcopy(H_dim, w+i*H_dim, 1, Q+i*H_dim, 1);
	   }

           for (int i = 0; i < unconv; i++) {
               cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+nroots)*H_dim, 1);
	   }
           gram_schmidt_orthogonalization(Q, nroots+unconv, H_dim);
	   L = nroots + unconv; 
	}
	else {
           for (int i = 0; i < unconv; i++) {
               cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+L)*H_dim, 1);
	   }
           gram_schmidt_add(Q, L, H_dim, unconv);
           L += unconv;	   
	}




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
    if (casscf == true) {
        //maxiter = 3;
	indim = nroots;
    }
    //printf(" maxiter%d", maxiter); 

    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    size_t H_dim = (N_p + 1) * num_alpha * num_alpha;
    
    double* Hdiag2 = (double*)malloc(H_dim*sizeof(double));
    cblas_dcopy(H_dim,Hdiag,1,Hdiag2,1);
    double* Q = (double*) malloc(maxdim*H_dim*sizeof(double));
    memset(Q, 0, maxdim*H_dim*sizeof(double));
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
    for (int a = 0; a < maxiter; a++) {
        printf("\n"); 
                
        printf("ITERATION%4d subspace size%4d\n", a+1, L);
        bool break_degeneracy = false;         
        memset(S, 0, maxdim*H_dim*sizeof(double));
        
	double itime, ftime, exec_time;
        itime = omp_get_wtime();

        build_sigma(h1e, h2e, d_cmo, Q, S, table, table_creation, table_annihilation, 
                        N_ac, n_o_ac, n_o_in, nmo, L, N_p, Enuc, dc, omega, d_exp, E_core, break_degeneracy); 

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
        //getMemory2(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);



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
            double dum = 0.0;
	    for (int i = 0; i < nroots; i++) {
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];	
		}
	    }
	    constdouble[4] = sqrt(dum);	
	    printf("converged\n");
    	    fflush(stdout);
	    break;
	}
	if ((a==maxiter-1) && (unconv > 0)) {
	    printf("Maximum iteration reaches. Please increase maxiter!\n");
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, eigenvecs, H_dim);
	    cblas_dcopy(nroots, theta, 1, eigenvals, 1);
            double dum = 0.0;
	    for (int i = 0; i < nroots; i++) {
                for (size_t j = 0; j < H_dim; j++) {
	 	    dum += w[i * H_dim +j] * w[i * H_dim +j];	
		}
	    }
	    constdouble[4] = sqrt(dum);	
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
	
        if (Lmax-L < unconv) {
           printf("maximum subspace reaches, restart!\n");		
           memset(w, 0, nroots*H_dim*sizeof(double));
           cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nroots, H_dim, L, 1.0, G, L, Q, H_dim, 0.0, w, H_dim);
           memset(Q, 0, maxdim*H_dim*sizeof(double));

           for (int i = 0; i < nroots; i++) {
               cblas_dcopy(H_dim, w+i*H_dim, 1, Q+i*H_dim, 1);
	   }

           for (int i = 0; i < unconv; i++) {
               cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+nroots)*H_dim, 1);
	   }
           gram_schmidt_orthogonalization(Q, nroots+unconv, H_dim);
	   L = nroots + unconv; 
	}
	else {
           for (int i = 0; i < unconv; i++) {
               cblas_dcopy(H_dim, w2+i*H_dim, 1, Q+(i+L)*H_dim, 1);
	   }
           gram_schmidt_add(Q, L, H_dim, unconv);
           L += unconv;	   
	}




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

void build_one_rdm(double* eigvec, double* D, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
    
    int n_occupied = n_o_ac + n_o_in;
    //double* D = (double*) malloc(nmo*nmo*sizeof(double));
    //memset(D, 0, nmo*nmo*sizeof(double));
    
    //inactive block
    for (int i = 0; i < n_o_in; i++) {
	int ii = i * n_occupied + i;
        D[ii] = 2.0 * (state_p1 == state_p2);	
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
        	    D[pq] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
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
        	    D[pq] += sign * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
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

void build_two_rdm(double* eigvec, double* D, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2) {
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




    double* temp1 = (double*) malloc(num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    memset(temp1, 0.0, num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    double* D_tu = (double*) malloc(n_o_ac*n_o_ac*sizeof(double));
    memset(D_tu, 0.0, n_o_ac*n_o_ac*sizeof(double));
    double* D_tuvw = (double*) malloc(n_o_ac*n_o_ac*n_o_ac*n_o_ac*sizeof(double));
    memset(D_tuvw, 0.0, n_o_ac*n_o_ac*n_o_ac*n_o_ac*sizeof(double));


    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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
        	    eigvec + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 0.0, D_tu, n_o_ac*n_o_ac);
    double* temp2 = (double*) malloc(num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    memset(temp2, 0.0, num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_o_ac*n_o_ac , n_o_ac*n_o_ac, num_photon*num_dets, 1.0, 
        	    temp2, n_o_ac*n_o_ac,
        	    temp1, n_o_ac*n_o_ac, 0.0, 
    		    D_tuvw, n_o_ac*n_o_ac);
    #pragma omp parallel for num_threads(16)
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
    free(temp1);
    free(temp2);
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
                    D[kl * n_occupied * n_occupied + ij] = (4.0 * (i==j) * (k==l) - 2.0 * (k==j) * (i==l)) * (state_p1 == state_p2);
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

void build_active_rdm(double* eigvec, double* D_tu, double* D_tuvw, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2, double weight) {
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    size_t num_dets = num_alpha * num_alpha;
   
    double* temp1 = (double*) malloc(num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    memset(temp1, 0.0, num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    double* temp0 = (double*) malloc(n_o_ac*n_o_ac*sizeof(double));
    memset(temp0, 0.0, n_o_ac*n_o_ac*sizeof(double));
   

    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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
        	    eigvec + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 0.0, temp0, n_o_ac*n_o_ac);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1 , n_o_ac*n_o_ac, num_photon*num_dets, weight, 
        	    eigvec + state_p1*num_photon*num_dets, num_photon*num_dets,
        	    temp1, n_o_ac*n_o_ac, 1.0, D_tu, n_o_ac*n_o_ac);
    double* temp2 = (double*) malloc(num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    memset(temp2, 0.0, num_dets*num_photon*n_o_ac*n_o_ac*sizeof(double));
    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(16)
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
    free(temp1);
    free(temp2);
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

void build_photon_electron_one_rdm(double* eigvec, double* Dpe, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2) {
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
        	    Dpe[ii] += 2.0 * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
		         * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_I];
        	    Dpe[ii] += 2.0 * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
		    	* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_I];
		}
                else if (photon_p == N_p) {
        	    Dpe[ii] += 2.0 * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
		         * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_I];
		}
                else{
        	    Dpe[ii] += 2.0 * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
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
        	        Dpe[pq] += sign * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe[pq] += sign * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe[pq] += sign * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe[pq] += sign * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
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
        	        Dpe[pq] += sign * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
        	        Dpe[pq] += sign * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
				* eigvec[(state_p2 * num_photon + photon_p + 1) * num_dets + index_J];
		    }
                    else if (photon_p == N_p) {
        	        Dpe[pq] += sign * sqrt(photon_p)   * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
			     * eigvec[(state_p2 * num_photon + photon_p - 1) * num_dets + index_J];
		    }
                    else{
        	        Dpe[pq] += sign * sqrt(photon_p+1) * eigvec[(state_p1 * num_photon + photon_p) * num_dets + index_I]
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

  
    #pragma omp parallel for num_threads(16)
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
    #pragma omp parallel for num_threads(12) collapse(2)
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
    //#pragma omp parallel for num_threads(12) collapse(2)
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
/* void getMemory2(
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
    printf("resident%20.12lf peak resident%20.12lf\n",(double)*currRealMem/1024.0/1024.0,(double)*peakRealMem/1024.0/1024.0);

} */
