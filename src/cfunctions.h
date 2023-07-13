

#include <stdint.h>
#include <stdbool.h>

double *h2e;
size_t num_alpha;
size_t N;
size_t n_o;
//size_t **graph;
//size_t **graph_big;
int **Y;
//int *table;
void matrix_product(double* A, double* B, double* C, int m,int n,int k);  
void printeverything(double *h2e, size_t num_alpha);
void get_graph(size_t N, size_t n_o, int** Y);
int* string_to_obtlist(size_t string, int nmo, int* length);
size_t index_to_string(int index, int N, int n_o, int** Y);    
void get_string (double* h1e, double* h2e, double* H_diag, int* table, int N_p, int num_alpha, int nmo, int N, int n_o, int n_in_a, double omega, double Enuc, double dc);  
void single_replacement_list(int num_alpha, int N, int n_o, int n_in_a, int** Y, int* table);
void build_H_diag(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha, int nmo, int n_act_a,int n_act_orb,int n_in_a, double omega, double Enuc, double dc); 
void sigma3(double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, size_t table_length, int nmo, int photon_p, int state_p, int num_photon);
void sigma12(double* h1e, double* h2e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, int photon_p, int state_p, int num_photon);  
void sigma_dipole(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon);
void constant_terms_contraction(double* c_vectors,double* c1_vectors,int num_alpha, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon);
void build_sigma(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors,
	       	int*table, size_t table_length, int num_links, int nmo, int num_alpha, int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, bool only_ground_state); 


