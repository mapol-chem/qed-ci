#include <stdlib.h>
#include <stdio.h>
#include "cfunctions2.h"
#include <string.h>
#include <math.h>
#include <cblas.h>

void matrix_product(double* A, double* B, double* C, int m, int n, int k) {
     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
}

void get_graph(size_t N, size_t n_o, int** Y) {
     //lexical ordering graph with unoccupied arc set to be zero, return vertex weight
     size_t rows = (N+1)*(n_o-N+1);
     size_t cols = 3;
     size_t rows1 = N+1;
     size_t cols1 = n_o+1;
     //size_t graph[rows][cols];
     //size_t graph_big[rows1][cols1];
     //graph = [[0 for i in range(cols)] for j in range(rows)]
     size_t graph[rows][cols];
     //graph_big = [[0 for i in range(cols1)] for j in range(rows1)]
     size_t graph_big[rows1][cols1];
     memset(graph_big, 0, sizeof graph_big);
     memset(graph, 0, sizeof graph);
     graph_big[N][n_o]=1;


     //weight of vertex
     for (int e = N; e>=0; e--) { 
         for (int o = n_o-1; o>=0; o--) {
             if (e==N && o>=e){
                 graph_big[e][o] = graph_big[e][o+1];
	     }
	     else if (e<=N && o<e){
                 graph_big[e][o] = 0;
	     }
             else{
                 graph_big[e][o] = graph_big[e+1][o+1] + graph_big[e][o+1];
	     }
	 }
     } 
     
     size_t count = 0;
     for (int e = 0; e<=N; e++) {
         for (int o = 0; o<=n_o; o++) {
             if (graph_big[e][o] !=0){
                 graph[count][0] = e;
                 graph[count][1] = o;
                 graph[count][2] = graph_big[e][o];
                 count +=1;
	     }
	 }
     }
     /*
    for (int i = 0; i<rows; i++) { 
         for (int j = 0; j<cols; j++) {
		 printf("%4d%4d%4d\n", i,j,graph[i][j]);
	 }
		 printf("\n");
     }
     fflush(stdout);
     */	   
     rows = N*(n_o-N+1);
       for (int row = 0; row < rows; row++) {
           //print(graph[i])
           int e = graph[row][0];   
           int o = graph[row][1];
           int B[1][3];
           if (e == N) {
               continue;
	   }
           int i = o - e;
           int c = 0;
           if (i == 0) {
               c = 0;
               //B.extend([e,o,c]);
	       B[0][0]=e;
	       B[0][1]=o;
	       B[0][2]=c;
               //printf("%4d%4d%4d\n",e,o,c);
	   }
           else {
               for (int j =1; j < i+1; j++) {
                   c += graph_big[e+1][o+2-j]; 
	       }
               B[0][0]=e;
	       B[0][1]=o;
	       B[0][2]=c;
               //printf("%4d%4d%4d\n",e,o,c);
	   }
           Y[row][0]=B[0][0];
           Y[row][1]=B[0][1];
           Y[row][2]=B[0][2];
       } 
       /*
       for (int i = 0; i < rows; i++) {
           for (int j = 0; j < cols; j++) {
	       printf("%4d%4d%4d\n",i,j,Y[i][j]);	   
	   }
		 printf("\n");
       }
       */
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
     //int b = n_o-sizeof(a)/sizeof(a[0]);
     //if (b>0) {
     //  for (int i = 0; i < b; i++) {
     //      b.pushback(0);
     //  }
     //}
       for (int i = 0; i < n_o; i++) {
	       printf("%d",a[i]);
       }
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







int string_to_index(size_t string, size_t N, size_t n_o, int** Y) {
     int a[n_o];
     memset(a, 0, sizeof a);
     int c=0;
     while(string > 0) {
	 size_t i = string%2;
         a[c++]=i; 		 
         string = (string-i)/2;  		     
     }
     int count = 0;
     int index = 0;
     int rows = N*(n_o-N+1);
     //for (int i = 0; i < n_o; i++) {
     //        printf("%4d",a[i]);
     //}
     //printf("\n");

     for (int i = 0; i < n_o; i++) {
         if (a[i] == 1) {
             int e = count;
             int o = i;
             for (int j = 0; j < rows; j++) {
                 if  (Y[j][0] == e && Y[j][1] == o) {
                     index +=Y[j][2];
		 }
	     }
             count +=1;
	 }	 
     } 
     return index;
}  
size_t index_to_string(int index, int N, int n_o, int** Y) {
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
                   if  (Y[j][0] == e-1 && Y[j][1] == o) {
                       int b = Y[j][2];
                       index_sum = index_sum-b;
		   }
	       }
               count3 -= count2;
	   }

           else {
               if (e > 0) {
                   for (int j = 0; j < rows; j++) {
                       if (Y[j][0] == e-1 && Y[j][1] == o-1) {
                           int a = Y[j][2];
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
               string += pow(2,i);
	   }
       }


return string;
   
}
int phase_single_excitation(size_t p, size_t q, size_t string) {
       size_t mask;
       if (p>q) {
           mask=(1<<p)-(1<<(q+1));
       }
       else {
           mask=(1<<q)-(1<<(p+1));
       }
       if (__builtin_popcount(mask & string) %2) {
           return -1;
       }
       else {
           return 1;
       }
}




void single_replacement_list(int num_alpha, int N, int n_o, int n_in_a, int** Y,int* table) {

       int count=0;        
       for (int index = 0; index < num_alpha; index++){
           size_t string = index_to_string(index,N,n_o,Y);
           int occ[N];
           int vir[n_o-N];
           int count_occ = 0;
           int count_vir = 0;
           for (int i = 0; i < n_o; i++) {
               if (string &(1<<i)) {
                   occ[count_occ] = i;
	           count_occ++;
	       }
               else {
                   vir[count_vir] = i;
		   count_vir++;
	       }
	   }
           if (n_in_a > 0) {
               for (int i = 0; i < n_in_a; i++) {
                   table[count*4+0] = index;
                   table[count*4+1] = 1; 
                   table[count*4+2] = i;
                   table[count*4+3] = i;
                   count += 1;
	       }
           }
           for (int i = 0; i < N; i++) {
               table[count*4+0] = index;
               table[count*4+1] = 1; 
               table[count*4+2] = occ[i] + n_in_a;
               table[count*4+3] = occ[i] + n_in_a;
               count += 1;
	   }
           for (int i = 0; i < N; i++) {
               for (int a = 0; a < n_o-N; a++) {
                   uint64_t string1 = (string^(1<<occ[i])) | (1<<vir[a]);
                   table[count*4+0] = string_to_index(string1,N,n_o,Y);
                   table[count*4+1] = phase_single_excitation(vir[a],occ[i],string);
                   table[count*4+2] = vir[a] + n_in_a;
                   table[count*4+3] = occ[i] + n_in_a;
                   count += 1;
	       }
	   }
       }
}

void build_H_diag(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha,int nmo, int n_act_a,int n_act_orb,int n_in_a, double omega, double Enuc, double dc) {
        size_t num_dets = num_alpha * num_alpha;
        int np1 = N_p + 1;
        for (int m = 0; m < np1; m++) {
            int start =  m * num_dets; 

            for (size_t Idet = 0; Idet < num_dets; Idet++) {
                int index_a = Idet/num_alpha;
                int index_b = Idet%num_alpha;
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
		//for (int i = 0; i < dim_d + 2*dim_s; i++) {
		//    for (int j = 0; j < 3; j++) {
		//        printf("%4d", occupation_list_spin[i*3+j]);
		//    }
		//        printf("\n");
	 	//    
		//}
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
                H_diag[Idet+start] = c + m * omega + Enuc + dc;
		free(double_active);
		free(single_occupation_a);
		free(single_occupation_b);
		free(occupation_list_spin);
	    }
	}

}


void get_string (double* h1e, double* h2e, double* H_diag, int* table, int N_p, int num_alpha, int nmo, int N, int n_o, int n_in_a, double omega, double Enuc, double dc){
     int rows = N*(n_o-N+1);
     int cols = 3;
     Y = (int**) malloc(rows*sizeof(int*));
     for (int row = 0; row < rows; row++) {
         Y[row] = (int*) malloc(cols*sizeof(int));
     }

     get_graph(N,n_o,Y);
     
     //int* p;
     //int length;
     //p=string_to_obtlist(23,10,&length);
     //for (int i = 0; i < length; i++) {
     //printf("%d\n", p[i]);
     //}
     //free(p);
       rows = num_alpha*(N*(n_o-N)+N+n_in_a);
       cols = 4;
       //size_t num_links = N*(n_o-N)+N+n_in_a
     
     build_H_diag(h1e, h2e, H_diag, N_p, num_alpha, nmo, N, n_o, n_in_a, omega, Enuc, dc);   
     //table = (int*) malloc(rows*cols*sizeof(int));
     single_replacement_list(num_alpha, N, n_o, n_in_a, Y, table);   
     //for (int i = 0; i < rows; i++) {
     //    for (int j = 0; j < cols; j++) {
     //        //table[i*cols+j] =1;
     //        printf("%4d", table[i*cols+j]);
     //    }
     //        printf("\n");
     //}

free(Y);
}
void build_sigma(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors, 
		 int*table, size_t table_length, int num_links, int nmo, int num_alpha, int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, bool only_ground_state) {


     int np1 = N_p + 1;
     //printf("total dim %d", num_alpha*num_alpha*num_state*np1);
     for (int n = 0; n < num_state; n++) {
         for (int m = 0; m < np1; m++) {
             sigma12(h1e, h2e, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, m, n, np1);  
             sigma3(h2e, c_vectors, c1_vectors, num_alpha, num_links, table, table_length, nmo, m, n, np1);  
             double someconstant = m * omega + Enuc + dc;
             if (only_ground_state == true) {
                someconstant = m * (omega + 1) + Enuc + dc;
	     }
             constant_terms_contraction(c_vectors, c1_vectors, num_alpha, someconstant, m, m, n, np1);    
             if ((0 < m) && (m < N_p)) {
                 someconstant = -sqrt(m * omega/2);
                 sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m-1, m, n, np1);  
                 constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);    
                 someconstant = -sqrt((m+1) * omega/2);
                 sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m+1, m, n, np1);  
                 constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);    
	     }
             else if (m == N_p) {
                 someconstant = -sqrt(m * omega/2);
                 sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m-1, m, n, np1);  
                 constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);   
	     }
             else {
                 someconstant = -sqrt((m+1) * omega/2);
                 sigma_dipole(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m+1, m, n, np1);  
                 constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);   
	     }
	 }
     }
}

       
void sigma3(double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, size_t table_length, int nmo, int photon_p, int state_p, int num_photon) {
     size_t num_dets = num_alpha * num_alpha;
     int* L = (int*) malloc(num_alpha*sizeof(int));
     int* R = (int*) malloc(num_alpha*sizeof(int));
     int* sgn = (int*) malloc(num_alpha*sizeof(int));
     double* cp;
     double* v;
     double* F;
     F = (double*) malloc(num_alpha*sizeof(double));
     memset(L, 0, num_alpha*sizeof(int));
     memset(R, 0, num_alpha*sizeof(int));
     memset(sgn, 0, num_alpha*sizeof(int));
      for (int k = 0; k < nmo; k++) {
          for (int l = 0; l < nmo; l++) {
              int kl = k * nmo + l;
      	      int dim = 0;
              for (size_t N = 0; N < table_length; N++) {
                  if ((table[N*4+2] == k) && (table[N*4+3] == l)) {
                      int index = N/num_links;
                      L[dim] = table[N*4+0];
                      R[dim] = index;
                      sgn[dim] = table[N*4+1];
                      dim +=1;
      	          }
      	      }
	      if (dim >0) {
              //print(L,R,sgn,dim)
                  cp = (double*) malloc(dim*num_alpha*sizeof(double));
                  memset(cp, 0, dim*num_alpha*sizeof(double));
                  for (int Ia = 0; Ia < dim; Ia++) {
                      for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
                          //int index_ljb = (photon_p * num_dets + L[Ia] * num_alpha + index_jb)*num_state+state_p;
                          int index_ljb = (state_p * num_photon + photon_p) * num_dets + L[Ia] * num_alpha + index_jb;
			  //printf("%4ld\n", index_ljb);
                          cp[Ia*num_alpha+index_jb] = c_vectors[index_ljb] * sgn[Ia];
	              }
	          }
                  for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                      memset(F, 0, num_alpha*sizeof(double));
                      int stride = index_ib * num_links;
                      for (int excitation = 0; excitation < num_links; excitation++) {
                          int index_jb = table[(stride + excitation)*4+0];
                          int sign = table[(stride + excitation)*4+1];
                          int i = table[(stride + excitation)*4+2]; 
                          int j = table[(stride + excitation)*4+3]; 
                          int ij = i * nmo + j;
                          F[index_jb] += sign * h2e[ij*nmo*nmo+kl];
	              }
                      v = (double*) malloc(dim*sizeof(double));
                      memset(v, 0, dim*sizeof(double));
                      //v = np.einsum("pq,q->p", cp, F)  
                      ////for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
                      ////    for (int Ia = 0; Ia < dim; Ia++) {
                      ////        v[Ia] += F[index_jb] * cp[Ia*num_alpha+index_jb];
	              ////    }
	              ////}
		      matrix_product(cp, F, v, dim,1,num_alpha);
                      for (int Ia = 0; Ia < dim; Ia++) {
                          int index_I = R[Ia] * num_alpha + index_ib;
                          c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += v[Ia];
	              } 
                      //free(v);     
	          }
                  //free(cp);     
	      }
          }
      }
free(R);     
free(L);     
free(sgn);     
free(F);     
free(v);     
free(cp);     
}
void sigma12(double* h1e, double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, int photon_p, int state_p, int num_photon) {
     size_t num_dets = num_alpha * num_alpha;
     double* F;
     F = (double*) malloc(num_alpha*sizeof(double));

     double* s_resize = (double*) malloc(num_alpha*sizeof(double));

     for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
         memset(F, 0, num_alpha*sizeof(double));
         int stride1 = index_ib * num_links;
         for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
             int index_kb = table[(stride1 + excitation1)*4+0];
             int sign1 = table[(stride1 + excitation1)*4+1];
             int k = table[(stride1 + excitation1)*4+2]; 
             int l = table[(stride1 + excitation1)*4+3]; 
             //print(index_kb,sign1,k,l)
             int kl = k * nmo + l;
             F[index_kb] += sign1 * h1e[k*nmo+l];
             int stride2 = index_kb * num_links;
             for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
                 int index_jb = table[(stride2 + excitation2)*4+0];
                 int sign2 = table[(stride2 + excitation2)*4+1];
                 int i = table[(stride2 + excitation2)*4+2]; 
                 int j = table[(stride2 + excitation2)*4+3]; 
                 int ij = i * nmo + j;
                 if (ij >= kl) {
                     F[index_jb] += (sign1 * sign2 * h2e[ij*nmo*nmo+kl])/(1+(ij == kl));
		 }
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
             //print(index_kb,sign1,k,l)
             int kl = k * nmo + l;
             F[index_ka] += sign1 * h1e[k*nmo+l];
             int stride2 = index_ka * num_links;
             for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
                 int index_ja = table[(stride2 + excitation2)*4+0];
                 int sign2 = table[(stride2 + excitation2)*4+1];
                 int i = table[(stride2 + excitation2)*4+2]; 
                 int j = table[(stride2 + excitation2)*4+3]; 
                 int ij = i * nmo + j;
                 if (ij >= kl) {
                     F[index_ja] += (sign1 * sign2 * h2e[ij*nmo*nmo+kl])/(1+(ij == kl));
		 }
	     }
	 }
         //vectorize ia
         //double* F2 = (double*) malloc(num_alpha*sizeof(double));
         //memset(F2, 0, num_alpha*sizeof(double));
         cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_alpha,1,num_alpha, 1.0, c_resize, num_alpha, F, 1, 1.0,
			 c1_vectors+(state_p * num_photon + photon_p) * num_dets+index_ia*num_alpha, 1);
         //matrix_product(c_resize, F, c1_vectors+(state_p * num_photon + photon_p) * num_dets + index_ia, num_alpha,1,num_alpha);


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
void sigma_dipole(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon) {

     size_t num_dets = num_alpha * num_alpha;
     double* F;
     F = (double*) malloc(num_alpha*sizeof(double));

     double* s_resize = (double*) malloc(num_alpha*sizeof(double));

     for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
         memset(F, 0, num_alpha*sizeof(double));
         int stride1 = index_ib * num_links;
         for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
             int index_kb = table[(stride1 + excitation1)*4+0];
             int sign1 = table[(stride1 + excitation1)*4+1];
             int k = table[(stride1 + excitation1)*4+2]; 
             int l = table[(stride1 + excitation1)*4+3]; 
             //print(index_kb,sign1,k,l)
             F[index_kb] += sign1 * h1e[k*nmo+l];
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
             //print(index_kb,sign1,k,l)
             F[index_ka] += sign1 * h1e[k*nmo+l];
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
     for (int index_I = 0; index_I < num_dets; index_I++) {
         c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += someconstant * c_vectors[(state_p * num_photon + photon_p1) * num_dets + index_I];
     }
}


