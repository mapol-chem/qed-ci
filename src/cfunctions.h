

#include <stdint.h>
#include <stdbool.h>

//double *h2e;
size_t num_alpha;
size_t N;
size_t n_o;
//size_t **graph;
//size_t **graph_big;
int *Y;

union Alpha_string {
    size_t string;
    int binary[0];
};

struct binary_output {
    int* binary;
    bool get_binary;
};
//int *table;
void matrix_product(double* A, double* B, double* C, int m,int n,int k);  
void get_graph(size_t N, size_t n_o, int* Y);
void get_graph_ras12(int n_o_ras1, int n_o_ras2, int N_ras1, int N_ras2, int* Y, bool shift_Y2);  
int* string_to_obtlist(size_t string, int nmo, int* length);
size_t index_to_string(int index, int N, int n_o, int* Y);    
//size_t index_to_string_ras12(int index, int n_o_ras1, int n_o_ras2, int N_ras1, int N_ras2, bool truncation); 
union Alpha_string* index_to_string_ras12(int index, int n_o_ras1, int n_o_ras2, int N_ras1, int N_ras2, bool truncation, bool return_string); 
struct binary_output index_to_binary2(int index, int N, int n_o, int* Y); 
int string_to_index_ras12(size_t string, int n_o_ras1, int n_o_ras2, int N);
void get_table(int N, int n_o_ras1, int n_o_ras2, int* table_creation,int* table_annihilation);
void get_string (double* h1e, double* h2e, double* H_diag, int* table, int N_p, int num_alpha, int nmo, int N, int n_o, int n_in_a, double omega, double Enuc, double dc);  
void single_replacement_list(int num_alpha, int N, int n_o, int n_in_a, int* Y, int* table);
void single_creation_list(int N, int n_o_ras1, int n_o_ras2, int* table_creation); 
void single_annihilation_list(int N, int n_o_ras1, int n_o_ras2, int* table_annihilation); 
void build_H_diag(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha, int nmo, int n_act_a,int n_act_orb,int n_in_a, double omega, double Enuc, double dc, int* Y); 
void sigma3(double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int table_length, int nmo, int photon_p, int state_p, int num_photon);
       
void sigma3_2(double* h2e, double* c_vectors, double* c1_vectors, int* table, int* table_creation, int* table_annihilation, 
		int* partition_index, int table_creation_length, int num_alpha, int n_o_ras1, int num_links, int num_links2, int nmo, int photon_p, int state_p, int num_photon);






void sigma12(double* h1e, double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, int photon_p, int state_p, int num_photon);  
void sigma_dipole(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon);
void constant_terms_contraction(double* c_vectors,double* c1_vectors,int num_alpha, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon);
void build_sigma(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors,
	       	int*table, int table_length, int num_links, int nmo, int num_alpha, int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, bool only_ground_state); 

void build_sigma_2(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors, 
		 int* table, int* table_creation, int* table_annihilation, int* partition_index, int table_creation_length, int num_links, int num_links2, int nmo, 
		 int num_alpha, int n_o_ras1, int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, bool only_ground_state); 


