#include <stdlib.h>
#include <stdio.h>
#include "cfunctions.h"
#include <string.h>
#include <math.h>
#include <cblas.h>
#include<omp.h>
#include<time.h>
//#include "memorymeasure.h"
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


void matrix_product(double* A, double* B, double* C, int m, int n, int k) {
     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
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



void get_graph2(int N, int n_o, int* Y, int scale) {
    
     // same as get_graph, but now we also consider subgraphs, so vertex weights are scaled by the number of paths going through intersection of two graphs
     //from bottom to intersection point
     int rows = (N+1)*(n_o-N+1);
     int cols = 3;
     int rows1 = N+1;
     int cols1 = n_o+1;
  
     int* graph = (int*) malloc(rows*cols*sizeof(int));
     memset(graph, 0, rows*cols*sizeof(int));

     //graph_big = [[0 for i in range(cols1)] for j in range(rows1)]
     //size_t graph_big[rows1][cols1];
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
     ////////for (int i = 0; i<rows; i++) { 
     ////////     for (int j = 0; j<cols; j++) {
     ////////    	 printf("%4d%4d%4d\n", i,j,graph[i*cols+j]);
     ////////     }
     ////////    	 printf("\n");
     ////////}
     //fflush(stdout);
     /*
     */	   
     rows = N*(n_o-N+1);
       for (int row = 0; row < rows; row++) {
           //print(graph[i])
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
               //B.extend([e,o,c]);
	       B[0][0]=e;
	       B[0][1]=o;
	       B[0][2]=c;
               //printf("%4d%4d%4d\n",e,o,c);
	   }
           else {
               for (int j =1; j < i+1; j++) {
                   c += scale * graph_big[(e+1)*cols1+o+2-j]; 
	       }
               B[0][0]=e;
	       B[0][1]=o;
	       B[0][2]=c;
               //printf("%4d%4d%4d\n",e,o,c);
	   }
           Y[row*3+0]=B[0][0];
           Y[row*3+1]=B[0][1];
           Y[row*3+2]=B[0][2];
       } 
       //for (int i = 0; i < rows; i++) {
       //    for (int j = 0; j < cols; j++) {
       //        printf("%4d%4d%4d\n",i,j,Y[i*3+j]);	   
       //    }
       // 	 printf("\n");
       //}
       /*
       */
free(graph);
free(graph_big);
}
void get_graph(size_t N, size_t n_o, int* Y) {
    
     //lexical ordering graph with unoccupied arc set to be zero, return vertex weight
     size_t rows = (N+1)*(n_o-N+1);
     size_t cols = 3;
     size_t rows1 = N+1;
     size_t cols1 = n_o+1;
     //size_t graph[rows][cols];
     //size_t graph_big[rows1][cols1];
     //graph = [[0 for i in range(cols)] for j in range(rows)]
     //size_t graph[rows][cols];

     int* graph = (int*) malloc(rows*cols*sizeof(int));
     memset(graph, 0, rows*cols*sizeof(int));

     //graph_big = [[0 for i in range(cols1)] for j in range(rows1)]
     //size_t graph_big[rows1][cols1];
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
     /*
     for (int i = 0; i<rows; i++) { 
         for (int j = 0; j<cols; j++) {
		 printf("%4d%4d%4d\n", i,j,graph[i*cols+j]);
	 }
		 printf("\n");
     }
     fflush(stdout);
     */	   
     rows = N*(n_o-N+1);
       for (int row = 0; row < rows; row++) {
           //print(graph[i])
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
               //B.extend([e,o,c]);
	       B[0][0]=e;
	       B[0][1]=o;
	       B[0][2]=c;
               //printf("%4d%4d%4d\n",e,o,c);
	   }
           else {
               for (int j =1; j < i+1; j++) {
                   c += graph_big[(e+1)*cols1+o+2-j]; 
	       }
               B[0][0]=e;
	       B[0][1]=o;
	       B[0][2]=c;
               //printf("%4d%4d%4d\n",e,o,c);
	   }
           Y[row*3+0]=B[0][0];
           Y[row*3+1]=B[0][1];
           Y[row*3+2]=B[0][2];
       } 
       /*
       for (int i = 0; i < rows; i++) {
           for (int j = 0; j < cols; j++) {
	       printf("%4d%4d%4d\n",i,j,Y[i*3+j]);	   
	   }
		 printf("\n");
       }
       */
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
    //}
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
                 if  (Y[j*3+0] == e && Y[j*3+1] == o) {
                     index +=Y[j*3+2];
		 }
	     }
             count +=1;
	 }	 
     } 
     return index;
} 


struct binary_output index_to_binary2(int index, int N, int n_o, int* Y) {
    struct binary_output str_out = {0};
        
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
        if (count == 1500000 || (e == N && o == N && index_sum<index)) {
            str_out.get_binary = false;
            break;
        }
        if ((e==0 && o==0 && index_sum==index)) {
            str_out.get_binary = true;
            break;
        }
    }
    if (str_out.get_binary == true) {
	str_out.binary = malloc(n_o * sizeof(int));    
        for (int i = 0; i < n_o; i++) {
           str_out.binary[i] = arr[i]; 
        }
    }


return str_out;
   
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
            if ((string &(1<<i)) == 0) {
                vir[count_vir] = i;
                size_t mask = (1<<i)-1;
		if (((__builtin_popcount(mask & string)) + n_o_in)  %2) {
		    sign[count_vir] = -1;
		}
		else {
		    sign[count_vir] = 1;
		}
     	        count_vir++;
            }
        }
        
        for (int a = 0; a < n_o_ac-N_ac+1; a++) {
            size_t string1 = string | (1<<vir[a]);
            table_creation[count*3+0] = string_to_index(string1,N_ac,n_o_ac,Y);
            table_creation[count*3+1] = sign[a];
            //table_creation[count*3+2] = vir[a] + n_o_in;
            table_creation[count*3+2] = vir[a];
            count += 1;
        }
    }
    free(Y1);
}

void single_annihilation_list2(int N_ac, int n_o_ac, int n_o_in, int* Y,int* table_annihilation) {
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
            if (string &(1<<i)) {
                occ[count_occ] = i;
     	        count_occ++;
            }
        }
	//printf("\n");
        
        for (int i = 0; i < N_ac; i++) {
            size_t string1 = string ^ (1<<occ[i]);
	    int sign;
	    if ((i+n_o_in)%2) {
	        sign = -1;
	    }
	    else {
		sign = 1;    
	    }
            table_annihilation[count*3+0] = string_to_index(string1,N_ac-1,n_o_ac,Y1);
            table_annihilation[count*3+1] = sign;
            //table_annihilation[count*3+2] = occ[i] + n_o_in;
            table_annihilation[count*3+2] = occ[i];
            count += 1;
        }
    }
    free(Y1);
}


void single_replacement_list2(int num_alpha, int N_ac, int n_o_ac, int* Y,int* table) {

       int count=0;        
       for (int index = 0; index < num_alpha; index++){
           size_t string = index_to_string(index,N_ac,n_o_ac,Y);
           int occ[N_ac];
           int vir[n_o_ac-N_ac];
           int count_occ = 0;
           int count_vir = 0;
           for (int i = 0; i < n_o_ac; i++) {
               if (string &(1<<i)) {
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
               table[count*4+2] = occ[i]; //+ n_o_in;
               table[count*4+3] = occ[i]; //+ n_o_in;
               count += 1;
	   }
           for (int i = 0; i < N_ac; i++) {
               for (int a = 0; a < n_o_ac-N_ac; a++) {
                   uint64_t string1 = (string^(1<<occ[i])) | (1<<vir[a]);
                   table[count*4+0] = string_to_index(string1,N_ac,n_o_ac,Y);
                   table[count*4+1] = phase_single_excitation(vir[a],occ[i],string);
                   table[count*4+2] = vir[a];// + n_o_in;
                   table[count*4+3] = occ[i];// + n_o_in;
                   count += 1;
	       }
	   }
       }
}

void single_replacement_list(int num_alpha, int N_ac, int n_o_ac, int n_o_in, int* Y,int* table) {

       int count=0;        
       for (int index = 0; index < num_alpha; index++){
           size_t string = index_to_string(index,N_ac,n_o_ac,Y);
           int occ[N_ac];
           int vir[n_o_ac-N_ac];
           int count_occ = 0;
           int count_vir = 0;
           for (int i = 0; i < n_o_ac; i++) {
               if (string &(1<<i)) {
                   occ[count_occ] = i;
	           count_occ++;
	       }
               else {
                   vir[count_vir] = i;
		   count_vir++;
	       }
	   }
           if (n_o_in > 0) {
               for (int i = 0; i < n_o_in; i++) {
                   table[count*4+0] = index;
                   table[count*4+1] = 1; 
                   table[count*4+2] = i;
                   table[count*4+3] = i;
                   count += 1;
	       }
           }
           for (int i = 0; i < N_ac; i++) {
               table[count*4+0] = index;
               table[count*4+1] = 1; 
               table[count*4+2] = occ[i] + n_o_in;
               table[count*4+3] = occ[i] + n_o_in;
               count += 1;
	   }
           for (int i = 0; i < N_ac; i++) {
               for (int a = 0; a < n_o_ac-N_ac; a++) {
                   uint64_t string1 = (string^(1<<occ[i])) | (1<<vir[a]);
                   table[count*4+0] = string_to_index(string1,N_ac,n_o_ac,Y);
                   table[count*4+1] = phase_single_excitation(vir[a],occ[i],string);
                   table[count*4+2] = vir[a] + n_o_in;
                   table[count*4+3] = occ[i] + n_o_in;
                   count += 1;
	       }
	   }
       }
}
void build_H_diag2(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha,int nmo, int N_ac,int n_o_ac,int n_o_in, double omega, double Enuc, double dc, int* Y) {
        //approximate diagonal elements for singlet roots
	
	size_t num_dets = num_alpha * num_alpha;
        int np1 = N_p + 1;
        
     //#pragma omp parallel for num_threads(16)
	for (size_t index_photon_det = 0; index_photon_det < np1*num_dets; index_photon_det++) {
	    size_t Idet = index_photon_det%num_dets;	
	    int m = (index_photon_det-Idet)/num_dets;	
            int start =  m * num_dets; 

            //for (size_t Idet = 0; Idet < num_dets; Idet++) {
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
		//for (int i = 0; i < dim_d + 2*dim_s; i++) {
		//        printf("%4d%4d%4d\n", occupation_list_spin[i*3+0], occupation_list_spin[i*3+1], occupation_list_spin[i*3+2]);
	 	//    
		//}
		fflush(stdout);
		double F = -1.0/(2*dim_s-1);
                double c = 0;
                for (int a = 0; a < (dim_d+2*dim_s); a++) {
                    int i = occupation_list_spin[a*3+0];
                    int n_ia = occupation_list_spin[a*3+1]; 
                    int n_ib = occupation_list_spin[a*3+2];
                    int n_i = n_ia+ n_ib; 
                    int N_i = n_i * (2 - n_i); 
                    int ii = i * nmo + i;
                    c += n_i * h1e[ii];
                    c -= 0.25 * N_i * h2e[ii*nmo+ii];
                    for (int b = 0; b < (dim_d+2*dim_s); b++) {
                        int j = occupation_list_spin[b*3+0];
                        int n_ja = occupation_list_spin[b*3+1]; 
                        int n_jb = occupation_list_spin[b*3+2];
                        int n_j = n_ja + n_jb; 
                        int N_j = n_j * (2 - n_j); 
                        int jj = j * nmo + j;
			int ij = i * nmo + j;
                        c += 0.5 * n_i * n_j * h2e[ii*nmo*nmo+jj];
                        c -= 0.25 * n_i * n_j * h2e[ij*nmo*nmo+ij];
                        if (i !=j) {
			    c -= 0.25 * F * N_i * N_j * h2e[ij*nmo*nmo+ij];
			}
		    }
		}
                H_diag[Idet+start] = c + m * omega + Enuc + dc;
	        //printf("%4d%20.12lf\n",2*dim_s, H_diag[Idet+start]);
		free(double_active);
		free(single_occupation_a);
		free(single_occupation_b);
		free(occupation_list_spin);
	    //printf("\n");
	}

}
void build_H_diag(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha,int nmo, int n_act_a,int n_act_orb,int n_in_a, double omega, double Enuc, double dc, int* Y) {
        size_t num_dets = num_alpha * num_alpha;
        int np1 = N_p + 1;
        
     #pragma omp parallel for num_threads(16)
	for (size_t index_photon_det = 0; index_photon_det < np1*num_dets; index_photon_det++) {
	    size_t Idet = index_photon_det%num_dets;	
	    int m = (index_photon_det-Idet)/num_dets;	
            int start =  m * num_dets; 

            //for (size_t Idet = 0; Idet < num_dets; Idet++) {
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
	    //}
	}

}


void get_string (double* h1e, double* h2e, double* H_diag, int* b_array, int* table, int* table1, int* table_creation, int* table_annihilation, int N_p, int num_alpha, int nmo, int N, int n_o, 
		int n_in_a, double omega, double Enuc, double dc, double target_spin){

/*
     int rows4 = 3 * (6 - 3 + 1) ;
     int* Y3 = (int*) malloc(rows4*3*sizeof(int));
     memset(Y3, 0, rows4*3*sizeof(int));
     get_graph(3,6,Y3);

int num_strings = binomialCoeff(6,2);
  int* table_creation = (int*) malloc(num_strings*4*3*sizeof(int));
     memset(table_creation, 0, num_strings*4*3*sizeof(int));

single_creation_list2(3,6, 2, Y3,table_creation);
      for (int row = 0; row <  num_strings*4; row++) {
              printf("%4d%4d%4d\n", table_creation[row*3+0], table_creation[row*3+1], table_creation[row*3+2]);
     }
 num_strings = binomialCoeff(6,3);
  int* table_annihilation = (int*) malloc(num_strings*3*3*sizeof(int));
     memset(table_annihilation, 0, num_strings*3*3*sizeof(int));

single_annihilation_list2(3,6, 2, Y3,table_annihilation);
      for (int row = 0; row <  num_strings*3; row++) {
              printf("%4d%4d%4d\n", table_annihilation[row*3+0], table_annihilation[row*3+1], table_annihilation[row*3+2]);
     }
 */



	
	
     //printf("\n");
     int rows = N*(n_o-N+1);
     int cols = 3;
     
     Y = (int*) malloc(rows*cols*sizeof(int));
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
     if (target_spin >=0) {
         build_H_diag2(h1e, h2e, H_diag, N_p, num_alpha, nmo, N, n_o, n_in_a, omega, Enuc, dc,Y);   
     }
     else {
         build_H_diag2(h1e, h2e, H_diag, N_p, num_alpha, nmo, N, n_o, n_in_a, omega, Enuc, dc,Y);   
     }
     ftime = omp_get_wtime();
     time_taken = ((double)t)/CLOCKS_PER_SEC;
     exec_time = ftime - itime;
     printf("buil_H_diag took %f seconds to execute \n", exec_time);

     t = clock();
     //printf("%4d%4d%4d",N,n_o,n_in_a);
     single_replacement_list(num_alpha, N, n_o, n_in_a, Y, table);   
     single_replacement_list2(num_alpha, N, n_o, Y, table1);   
     single_creation_list2(N,n_o, n_in_a, Y,table_creation);
     single_annihilation_list2(N,n_o, n_in_a, Y,table_annihilation);
     t = clock() - t;
     time_taken = ((double)t)/CLOCKS_PER_SEC;
     printf("single_replacement_list took %f seconds to execute \n", time_taken);


     //int* b_array = (int*) malloc(num_alpha*n_o*n_o*2*sizeof(int));
     //memset(b_array, -1, num_alpha*n_o*n_o*2*sizeof(int));
     int num_links = N * (n_o-N) + N;
     build_b_array(table1, b_array, num_alpha, num_links, n_o);
     ////for (int i = 0; i < num_alpha; i++) {
     ////    for (int j = 0; j < n_o*n_o; j++) {
     ////        int q = j%n_o;
     ////        int p = (j-q)/n_o; 	     
     ////        printf("%4d%4d%4d%4d%4d\n",i,p,q,b_array[i*n_o*n_o*2+j*2+0],b_array[i*n_o*n_o*2+j*2+1]);		 
     ////    }
     ////}

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
		 int*table, int table_length, int num_links, int nmo, int num_alpha, int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, bool break_degeneracy) {


     //int threads =omp_get_max_threads();
     //printf(" numthread %d",threads);
     //  #pragma omp parallel for num_threads(12)
     // for (int i = 1; i <= 12; i++) {
     //     int tid = omp_get_thread_num();
     //     printf("The thread %d  executes i = %d\n", tid, i);
     // }



     int np1 = N_p + 1;
     //printf("total dim %d", num_alpha*num_alpha*num_state*np1);
     #pragma omp parallel for num_threads(12) collapse(2)
     for (int n = 0; n < num_state; n++) {
         for (int m = 0; m < np1; m++) {
             sigma12(h1e, h2e, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, m, n, np1); 
             sigma3(h2e, c_vectors, c1_vectors, num_alpha, num_links, table, table_length, nmo, m, n, np1);  
             double someconstant = m * omega + Enuc + dc;
             if (break_degeneracy == true) {
                someconstant = m * (omega + 1) + Enuc + dc;
	     }
             constant_terms_contraction(c_vectors, c1_vectors, num_alpha, someconstant, m, m, n, np1);   
	     if (N_p == 0) continue;
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


void build_sigma_3(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors, 
		 int* table,int* table1, int* table_creation, int* table_annihilation, int N_ac, int n_o_ac, int n_o_in, int nmo, 
		 int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, bool break_degeneracy) {


    //int threads =omp_get_max_threads();
    //printf(" numthread %d",threads);
    //  #pragma omp parallel for num_threads(12)
    // for (int i = 1; i <= 12; i++) {
    //     int tid = omp_get_thread_num();
    //     printf("The thread %d  executes i = %d\n", tid, i);
    // }

    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac + n_o_in;
    int np1 = N_p + 1;
    //printf("total dim %d", num_alpha*num_alpha*num_state*np1);
    #pragma omp parallel for num_threads(12) collapse(2)
    for (int n = 0; n < num_state; n++) {
        for (int m = 0; m < np1; m++) {
            sigma12(h1e, h2e, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, m, n, np1); 
            sigma3_3(h2e, c_vectors, c1_vectors, table1, table_creation, table_annihilation, 
       	   N_ac, n_o_ac, n_o_in, nmo, m, n, np1);

            double someconstant = m * omega + Enuc + dc;
            if (break_degeneracy == true) {
               someconstant = m * (omega + 1) + Enuc + dc;
            }
            constant_terms_contraction(c_vectors, c1_vectors, num_alpha, someconstant, m, m, n, np1);    
	    if (N_p == 0) continue;
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
       
void sigma3(double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int table_length, int nmo, int photon_p, int state_p, int num_photon) {
     //printf("he3\n");
     //unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
     //int success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);


    
     size_t num_dets = num_alpha * num_alpha;
     int* L = (int*) malloc(num_alpha*sizeof(int));
     int* R = (int*) malloc(num_alpha*sizeof(int));
     int* sgn = (int*) malloc(num_alpha*sizeof(int));
     double* F = (double*) malloc(num_alpha*sizeof(double));
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
                  double* cp;
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
                      double* v;
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
                      free(v);     
	          }
                  free(cp);     
	      }
          }
      }
free(R);     
free(L);     
free(sgn);     
free(F);     
//free(v);     
//free(cp); 
     //success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
     //printf("he4\n");
   
}

void sigma3_3(double* h2e, double* c_vectors, double* c1_vectors, int* table1,  int* table_creation, int* table_annihilation, 
		 int N_ac, int n_o_ac, int n_o_in, int nmo, int photon_p, int state_p, int num_photon) {
    //printf("he3\n");
    //unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    //int success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_alpha1 = binomialCoeff(n_o_ac, N_ac-1);
    size_t num_dets = num_alpha * num_alpha;
    
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    int num_links1 = n_o_ac-N_ac+1;
    int num_links2 = N_ac;

    //printf("%4d%4d%4d%4d%4d\n",N_ac,n_o_ac, n_o_in,nmo, num_links);
    double* D = (double*) malloc(num_alpha1 * n_o_ac * num_alpha*sizeof(double));
    memset(D, 0, num_alpha1 * n_o_ac * num_alpha * sizeof(double));
    for (int index_ka = 0; index_ka < num_alpha1; index_ka++) {
	int stride = index_ka * num_links1;    	
        for (int creation = 0; creation < num_links1; creation++) {
	    int index_ja = table_creation[(stride+creation)*3+0];
	    int sign = table_creation[(stride+creation)*3+1];
	    int j = table_creation[(stride+creation)*3+2];
	    //printf("%4d%4d%4d\n",index_ja,sign,j);
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
	        int index_J = index_ja * num_alpha + index_jb;
                D[(index_jb * n_o_ac + j) * num_alpha1 + index_ka] += sign *
                     c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];


	    }
	}
		
    }

    //printf("\n");
    double* T = (double*) malloc(num_alpha1 * n_o_ac * num_alpha*sizeof(double));
    memset(T, 0, num_alpha1 * n_o_ac * num_alpha * sizeof(double));
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        //memset(F, 0, num_alpha*sizeof(double));
        int stride = index_ib * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table1[(stride + excitation)*4+0];
            int sign = table1[(stride + excitation)*4+1];
            int k = table1[(stride + excitation)*4+2]; 
            int l = table1[(stride + excitation)*4+3]; 
	    //printf("%4d%4d%4d%4d%10d\n",index_jb,sign,k,l,index_ib);
            int kl = (k+n_o_in) * nmo + (l+n_o_in);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_o_ac, num_alpha1, n_o_ac, sign, h2e+kl*nmo*nmo+n_o_in*nmo+n_o_in, 
               	  nmo, D+index_jb*n_o_ac*num_alpha1, num_alpha1, 1.0, 
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
            //printf("%4d%4d%4d\n",index_ka,sign,i);
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                int index_I = index_ia * num_alpha + index_ib;
                c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign *
                T[(index_ib * n_o_ac + i) * num_alpha1 + index_ka];
            }
        }
        	
    }
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        int stride = index_ib * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table1[(stride + excitation)*4+0];
            int sign = table1[(stride + excitation)*4+1];
            int k = table1[(stride + excitation)*4+2]; 
            int l = table1[(stride + excitation)*4+3]; 
            //print(index_kb,sign1,k,l)
            int kl = (k+n_o_in) * nmo + (l+n_o_in);
	    double c = 0.0;
            for (int i = 0; i < n_o_in; i++) {
		int ii = i * nmo + i;    
		c += h2e[ii*nmo*nmo + kl];   
	    }
            for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
                int index_I = index_ia * num_alpha + index_ib;
                int index_J = index_ia * num_alpha + index_jb;
                //for (int i = 0; i < n_o_in; i++) {
	        //    int ii = i * nmo + i;    
                  c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign *
	          	/*h2e[ii*nmo*nmo + kl]*/ c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];
	        //}
            } 
        }
    }
    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
        int stride = index_ia * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ja = table1[(stride + excitation)*4+0];
            int sign = table1[(stride + excitation)*4+1];
            int k = table1[(stride + excitation)*4+2]; 
            int l = table1[(stride + excitation)*4+3]; 
            //print(index_kb,sign1,k,l)
            int kl = (k+n_o_in) * nmo + (l+n_o_in);
	    double c = 0.0;
            for (int i = 0; i < n_o_in; i++) {
		int ii = i * nmo + i;    
		c += h2e[ii*nmo*nmo + kl];   
	    }
            for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
                int index_I = index_ia * num_alpha + index_ib;
                int index_J = index_ja * num_alpha + index_ib;
                //for (int i = 0; i < n_o_in; i++) {
		//    int ii = i * nmo + i;    
                    c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign * 
    			 /*h2e[ii*nmo*nmo + kl]*/ c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];
	        //}
	       
            } 
        }
    }




    double c = 0.0;
    for (int i = 0; i < n_o_in; i++) {
        for (int j = 0; j < n_o_in; j++) {
            int ii = i * nmo +i; 
            int jj = j * nmo +j; 
            c += h2e[ii*nmo*nmo + jj];
	}
    }
    for (size_t index_I = 0; index_I < num_dets; index_I++) {
        c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_I];
    }

//fflush(stdout);
    
//success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
//printf("he4\n");
//fflush(stdout);
free(D);
free(T);

}





void sigma12(double* h1e, double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, int photon_p, int state_p, int num_photon) {
    //printf("he1\n");
    //unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    //int success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);


    size_t num_dets = num_alpha * num_alpha;
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
    //success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    //printf("he2\n");


    
}
void sigma_dipole(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon) {
     //printf("he5\n");
     //unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
     //int success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);


     size_t num_dets = num_alpha * num_alpha;
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
     //success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
     //printf("he6\n");


}
void constant_terms_contraction(double* c_vectors,double* c1_vectors,int num_alpha, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon) {
     size_t num_dets = num_alpha * num_alpha;
     for (size_t index_I = 0; index_I < num_dets; index_I++) {
         c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += someconstant * c_vectors[(state_p * num_photon + photon_p1) * num_dets + index_I];
     }
}

//int main() {
//	int N = 3;
//	int n_o = 5;
//    int rows = N*(n_o-N+1);
//    int cols = 3;
//
//    Y = (int*) malloc(rows*cols*sizeof(int));
//    get_graph(N,n_o,Y);
//    int* table_creation;
//    int length;
//    table_creation = single_creation_list(5, 2,5, &length);
//
//    free(Y);
//}
void build_b_array(int* table1, int* b_array, int num_alpha, int num_links, int n_o_ac) {
    //int n_pq = nmo * (nmo-1)/2;	
    //for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    //    int stride = index_ib * num_links;
    //    for (int pq = 0; pq < n_pq; pq++) {
    //        int p = nmo - 2 - floor(sqrt(-8*pq + 4*nmo*(nmo-1)-7)/2.0 - 0.5);
    //        int q = pq + p + 1 - n_pq + (nmo-p) * ((nmo-p)-1)/2;
    //        for (int excitation = 0; excitation < num_links; excitation++) {
    //            if (((table[(stride+excitation)*4+2] == p) && (table[(stride+excitation)*4+3] == q)) ||
    //    		       	((table[(stride+excitation)*4+2] == q) && (table[(stride+excitation)*4+3] == p))) {
    //                b_matrix[index_ib * n_pq + pq] = table[(stride+excitation)*4+0];
    //            }
    //  	    }
    //    }
    //}
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
    
    Y = (int*) malloc(rows*cols*sizeof(int));
    get_graph(N_ac,n_o_ac,Y);

  
     //#pragma omp parallel for num_threads(16)
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
    	//for (int i = 0; i < dim_d + 2*dim_s; i++) {
    	//    printf("%4d%4d%4d\n", occupation_list_spin[i*3+0], occupation_list_spin[i*3+1], occupation_list_spin[i*3+2]);
    	//}
    	//fflush(stdout);
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
    //printf("total dim %d", num_alpha*num_alpha*num_state*np1);
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
void build_sigma_spin(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors, 
		 int* table,int* table1, int* table_creation, int* table_annihilation, int N_ac, int n_o_ac, int n_o_in, int nmo, 
		 int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, bool break_degeneracy, double target_spin) {


    //int threads =omp_get_max_threads();
    //printf(" numthread %d",threads);
    //  #pragma omp parallel for num_threads(12)
    // for (int i = 1; i <= 12; i++) {
    //     int tid = omp_get_thread_num();
    //     printf("The thread %d  executes i = %d\n", tid, i);
    // }

    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac + n_o_in;
    int np1 = N_p + 1;
    //printf("total dim %d", num_alpha*num_alpha*num_state*np1);
    #pragma omp parallel for num_threads(12) collapse(2)
    for (int n = 0; n < num_state; n++) {
        for (int m = 0; m < np1; m++) {
            sigma12_spin(h1e, h2e, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, m, n, np1, target_spin); 
            sigma3_spin(h2e, c_vectors, c1_vectors, table1, table_creation, table_annihilation, 
       	   N_ac, n_o_ac, n_o_in, nmo, m, n, np1, target_spin);

            double someconstant = m * omega + Enuc + dc;
            if (break_degeneracy == true) {
               someconstant = m * (omega + 1) + Enuc + dc;
            }
            constant_terms_contraction(c_vectors, c1_vectors, num_alpha, someconstant, m, m, n, np1);    
	    if (N_p == 0) continue;
            if ((0 < m) && (m < N_p)) {
                someconstant = -sqrt(m * omega/2);
                sigma_dipole_spin(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m-1, m, n, np1, target_spin);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);    
                someconstant = -sqrt((m+1) * omega/2);
                sigma_dipole_spin(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m+1, m, n, np1, target_spin);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);    
            }
            else if (m == N_p) {
                someconstant = -sqrt(m * omega/2);
                sigma_dipole_spin(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m-1, m, n, np1, target_spin);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m-1, m, n, np1);   
            }
            else {
                someconstant = -sqrt((m+1) * omega/2);
                sigma_dipole_spin(d_cmo, c_vectors, c1_vectors, num_alpha, num_links, table, nmo, someconstant, m+1, m, n, np1, target_spin);  
                constant_terms_contraction(c_vectors, c1_vectors, num_alpha, -d_exp * someconstant, m+1, m, n, np1);   
            }
        }
    }
}

void sigma3_spin(double* h2e, double* c_vectors, double* c1_vectors, int* table1,  int* table_creation, int* table_annihilation, 
		 int N_ac, int n_o_ac, int n_o_in, int nmo, int photon_p, int state_p, int num_photon, double target_spin) {
    //printf("he3\n");
    //unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    //int success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    int num_alpha = binomialCoeff(n_o_ac, N_ac);
    int num_alpha1 = binomialCoeff(n_o_ac, N_ac-1);
    size_t num_dets = num_alpha * num_alpha;
    int num_links = N_ac * (n_o_ac-N_ac) + N_ac;
    int num_links1 = n_o_ac-N_ac+1;
    int num_links2 = N_ac;
    int length_sigma3 = num_alpha * num_alpha;
    double* sigma3 = (double*) malloc(length_sigma3*sizeof(double));
    memset(sigma3, 0, length_sigma3*sizeof(double));


    //printf("%4d%4d%4d%4d%4d\n",N_ac,n_o_ac, n_o_in,nmo, num_links);
    double* D = (double*) malloc(num_alpha1 * n_o_ac * num_alpha*sizeof(double));
    memset(D, 0, num_alpha1 * n_o_ac * num_alpha * sizeof(double));
    for (int index_ka = 0; index_ka < num_alpha1; index_ka++) {
	int stride = index_ka * num_links1;    	
        for (int creation = 0; creation < num_links1; creation++) {
	    int index_ja = table_creation[(stride+creation)*3+0];
	    int sign = table_creation[(stride+creation)*3+1];
	    int j = table_creation[(stride+creation)*3+2];
	    //printf("%4d%4d%4d\n",index_ja,sign,j);
            for (int index_jb = 0; index_jb < num_alpha; index_jb++) {
	        int index_J = index_ja * num_alpha + index_jb;
                D[(index_jb * n_o_ac + j) * num_alpha1 + index_ka] += sign *
                     c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];


	    }
	}
		
    }

    //printf("\n");
    double* T = (double*) malloc(num_alpha1 * n_o_ac * num_alpha*sizeof(double));
    memset(T, 0, num_alpha1 * n_o_ac * num_alpha * sizeof(double));
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        //memset(F, 0, num_alpha*sizeof(double));
        int stride = index_ib * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table1[(stride + excitation)*4+0];
            int sign = table1[(stride + excitation)*4+1];
            int k = table1[(stride + excitation)*4+2]; 
            int l = table1[(stride + excitation)*4+3]; 
	    //printf("%4d%4d%4d%4d%10d\n",index_jb,sign,k,l,index_ib);
            int kl = (k+n_o_in) * nmo + (l+n_o_in);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_o_ac, num_alpha1, n_o_ac, sign, h2e+kl*nmo*nmo+n_o_in*nmo+n_o_in, 
               	  nmo, D+index_jb*n_o_ac*num_alpha1, num_alpha1, 1.0, 
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
            //printf("%4d%4d%4d\n",index_ka,sign,i);
            for (int index_ib = index_ia; index_ib < num_alpha; index_ib++) {
                int index_I = index_ia * num_alpha + index_ib;
                //c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign *
                //T[(index_ib * n_o_ac + i) * num_alpha1 + index_ka];
		sigma3[index_I] += sign * T[(index_ib * n_o_ac + i) * num_alpha1 + index_ka];

            }
        }
        	
    }
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        int stride = index_ib * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_jb = table1[(stride + excitation)*4+0];
            int sign = table1[(stride + excitation)*4+1];
            int k = table1[(stride + excitation)*4+2]; 
            int l = table1[(stride + excitation)*4+3]; 
            //print(index_kb,sign1,k,l)
            int kl = (k+n_o_in) * nmo + (l+n_o_in);
	    double c = 0.0;
            for (int i = 0; i < n_o_in; i++) {
		int ii = i * nmo + i;    
		c += h2e[ii*nmo*nmo + kl];   
	    }
            for (int index_ia = 0; index_ia <= index_ib; index_ia++) {
                int index_I = index_ia * num_alpha + index_ib;
                int index_J = index_ia * num_alpha + index_jb;
                //for (int i = 0; i < n_o_in; i++) {
	        //    int ii = i * nmo + i;    
                 //// c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign *
	         //// 	 c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];
	        sigma3[index_I] += sign * c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];

		//}
            } 
        }
    }
    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
        int stride = index_ia * num_links;
        for (int excitation = 0; excitation < num_links; excitation++) {
            int index_ja = table1[(stride + excitation)*4+0];
            int sign = table1[(stride + excitation)*4+1];
            int k = table1[(stride + excitation)*4+2]; 
            int l = table1[(stride + excitation)*4+3]; 
            //print(index_kb,sign1,k,l)
            int kl = (k+n_o_in) * nmo + (l+n_o_in);
	    double c = 0.0;
            for (int i = 0; i < n_o_in; i++) {
		int ii = i * nmo + i;    
		c += h2e[ii*nmo*nmo + kl];   
	    }
            for (int index_ib = index_ia; index_ib < num_alpha; index_ib++) {
                int index_I = index_ia * num_alpha + index_ib;
                int index_J = index_ja * num_alpha + index_ib;
                //for (int i = 0; i < n_o_in; i++) {
		//    int ii = i * nmo + i;    
                   //// c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sign * 
    			////  c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];
		sigma3[index_I] += sign * c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_J];

	        //}
	       
            } 
        }
    }
    //symmetrize/antisymmetrize CI coefficient if spin is even/odd 
    for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
        for (int index_ib = index_ia; index_ib < num_alpha; index_ib++) {
            int index_I1 = index_ia * num_alpha + index_ib;
            int index_I2 = index_ib * num_alpha + index_ia;
            c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I1] += sigma3[index_I1] / (1+(index_ia == index_ib));
            c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I2] += pow(-1.0, target_spin) * sigma3[index_I1] / (1+(index_ia == index_ib));
	}
    }


    double c = 0.0;
    for (int i = 0; i < n_o_in; i++) {
        for (int j = 0; j < n_o_in; j++) {
            int ii = i * nmo +i; 
            int jj = j * nmo +j; 
            c += h2e[ii*nmo*nmo + jj];
	}
    }
    for (size_t index_I = 0; index_I < num_dets; index_I++) {
        c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += c * c_vectors[(state_p * num_photon + photon_p) * num_dets + index_I];
    }

//fflush(stdout);
    
//success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
//printf("he4\n");
//fflush(stdout);
free(sigma3);
free(D);
free(T);

}







void sigma12_spin(double* h1e, double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, int photon_p, int state_p, int num_photon, double target_spin) {
    //printf("he1\n");
    //unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    //int success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);

    size_t num_dets = num_alpha * num_alpha;
    double* F = (double*) malloc(num_alpha*sizeof(double));

    double* s_resize = (double*) malloc(num_alpha*sizeof(double));
    double* sigma1 = (double*) malloc(num_dets*sizeof(double));
    memset(sigma1, 0, num_dets*sizeof(double));

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
            //c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += s_resize[index_ia];
            sigma1[index_I] += s_resize[index_ia];
        }
    }
    //symmetrize/antisymmetrize CI coefficient if spin is even/odd 
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
            int index_I = index_ia * num_alpha + index_ib;
            c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += sigma1[index_ia * num_alpha + index_ib];
            c1_vectors[(state_p * num_photon + photon_p) * num_dets + index_I] += pow(-1.0, target_spin) * sigma1[index_ib * num_alpha + index_ia];
	}
    }


    ////double* c_resize = (double*) malloc(num_dets*sizeof(double));
    ////
    ////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////        int index_J = index_ja * num_alpha + index_ib;
    ////        int index_Jt = index_ib * num_alpha + index_ja;
    ////        c_resize[index_Jt] = c_vectors[
    ////   	 (state_p * num_photon + photon_p) * num_dets + index_J];
    ////    }
    ////}
    ////
    ////
    ////
    ////for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////    memset(F, 0, num_alpha*sizeof(double));
    ////    int stride1 = index_ia * num_links;
    ////    for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
    ////        int index_ka = table[(stride1 + excitation1)*4+0];
    ////        int sign1 = table[(stride1 + excitation1)*4+1];
    ////        int k = table[(stride1 + excitation1)*4+2]; 
    ////        int l = table[(stride1 + excitation1)*4+3]; 
    ////        //print(index_kb,sign1,k,l)
    ////        int kl = k * nmo + l;
    ////        F[index_ka] += sign1 * h1e[k*nmo+l];
    ////        int stride2 = index_ka * num_links;
    ////        for (int excitation2 = 0; excitation2 < num_links; excitation2++) {
    ////            int index_ja = table[(stride2 + excitation2)*4+0];
    ////            int sign2 = table[(stride2 + excitation2)*4+1];
    ////            int i = table[(stride2 + excitation2)*4+2]; 
    ////            int j = table[(stride2 + excitation2)*4+3]; 
    ////            int ij = i * nmo + j;
    ////            if (ij >= kl) {
    ////                F[index_ja] += (sign1 * sign2 * h2e[ij*nmo*nmo+kl])/(1+(ij == kl));
    ////   	        }
    ////        }
    ////    }
    ////    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_alpha,1,num_alpha, 1.0, c_resize, num_alpha, F, 1, 1.0,
    ////   		 c1_vectors+(state_p * num_photon + photon_p) * num_dets+index_ia*num_alpha, 1);

    ////}
    free(F); 
    free(sigma1); 
    //free(c_resize); 
    free(s_resize);
    //success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    //printf("he2\n");


    
}
void sigma_dipole_spin(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, double someconstant, int photon_p1, int photon_p2, 
		int state_p, int num_photon, double target_spin) {
    //printf("he5\n");
    //unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    //int success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);


    size_t num_dets = num_alpha * num_alpha;
    double* F = (double*) malloc(num_alpha*sizeof(double));
    double* sigma1 = (double*) malloc(num_dets*sizeof(double));
    memset(sigma1, 0, num_dets*sizeof(double));


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
            sigma1[index_I] += s_resize[index_ia];
        }
    }
    //symmetrize/antisymmetrize CI coefficient if spin is even/odd 
    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
        for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
            int index_I = index_ia * num_alpha + index_ib;
            c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += sigma1[index_ia * num_alpha + index_ib];
            c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += pow(-1.0, target_spin) * sigma1[index_ib * num_alpha + index_ia];
	}
    }
    ////double* c_resize = (double*) malloc(num_dets*sizeof(double));
    ////
    ////for (int index_ja = 0; index_ja < num_alpha; index_ja++) {
    ////    for (int index_ib = 0; index_ib < num_alpha; index_ib++) {
    ////        int index_J = index_ja * num_alpha + index_ib;
    ////        int index_Jt = index_ib * num_alpha + index_ja;
    ////        c_resize[index_Jt] = c_vectors[
    ////   	 (state_p * num_photon + photon_p1) * num_dets + index_J];
    ////    }
    ////}

    ////for (int index_ia = 0; index_ia < num_alpha; index_ia++) {
    ////    memset(F, 0, num_alpha*sizeof(double));
    ////    int stride1 = index_ia * num_links;
    ////    for (int excitation1 = 0; excitation1 < num_links; excitation1++) {
    ////        int index_ka = table[(stride1 + excitation1)*4+0];
    ////        int sign1 = table[(stride1 + excitation1)*4+1];
    ////        int k = table[(stride1 + excitation1)*4+2]; 
    ////        int l = table[(stride1 + excitation1)*4+3]; 
    ////        //print(index_kb,sign1,k,l)
    ////        F[index_ka] += sign1 * h1e[k*nmo+l];
    ////    }
    ////    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, num_alpha,1,num_alpha, someconstant, c_resize, num_alpha, F, 1, 1.0,
    ////   		 c1_vectors+(state_p * num_photon + photon_p2) * num_dets+index_ia*num_alpha, 1);
    ////}
    free(sigma1);
    free(F); 
    //free(c_resize); 
    free(s_resize); 
    //success2 = getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    //printf("he6\n");


}
void constant_terms_contraction_spin(double* c_vectors,double* c1_vectors,int num_alpha, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon) {
     size_t num_dets = num_alpha * num_alpha;
     for (size_t index_I = 0; index_I < num_dets; index_I++) {
         c1_vectors[(state_p * num_photon + photon_p2) * num_dets + index_I] += someconstant * c_vectors[(state_p * num_photon + photon_p1) * num_dets + index_I];
     }
}



