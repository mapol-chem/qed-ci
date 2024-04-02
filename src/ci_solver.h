#include <stdint.h>
#include <stdbool.h>
typedef void (*callback_)(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors, 
		 int* table, int* table_creation, int* table_annihilation, int N_ac, int n_o_ac, int n_o_in, int nmo, 
		 int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, double E_core, bool break_degeneracy);


//int *Y;
//int *table;
void matrix_product(double* A, double* B, double* C, int m,int n,int k);  
void get_graph(size_t N, size_t n_o, int* Y);
int* string_to_obtlist(size_t string, int nmo, int* length);
size_t index_to_string(int index, int N, int n_o, int* Y);    
void get_string(double* h1e, double* h2e, double* H_diag, int* b_array, int* table, int* table_creation, 
		int* table_annihilation, int N_p, int num_alpha, int nmo, int N, int n_o, int n_in_a, double omega, double Enuc, double dc);  
void single_replacement_list(int num_alpha, int N_ac, int n_o_ac, int* Y, int* table);
void build_H_diag(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha, int nmo, int n_act_a,int n_act_orb,int n_in_a, double omega, double Enuc, double dc, int* Y); 
void sigma3(double* h2e, double* c_vectors, double* c1_vectors, int* table, int* table_creation, int* table_annihilation, 
		 int N_ac, int n_o_ac, int n_o_in, int nmo, int photon_p, int state_p, int num_photon); 


void sigma12(double* h1e, double* h2e, double* c_vectors, double* c1_vectors, int num_alpha, int num_links, int* table, int nmo, int n_o_ac, int n_o_in,
	       	int photon_p, int state_p, int num_photon);  
void sigma_dipole(double* h1e, double* c_vectors,double* c1_vectors,int num_alpha,int num_links, int* table, int nmo, int n_o_in,
		 double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon);
void constant_terms_contraction(double* c_vectors,double* c1_vectors,int num_alpha, double someconstant, int photon_p1, int photon_p2, int state_p, int num_photon);
void build_sigma(double* h1e, double* h2e, double* d_cmo, double* c_vectors, double *c1_vectors, 
		 int* table, int* table_creation, int* table_annihilation, int N_ac, int n_o_ac, int n_o_in, int nmo, 
		 int num_state, int N_p, double Enuc, double dc, double omega, double d_exp, double E_core, bool break_degeneracy);

void build_b_array(int* table, int* b_matrix, int num_alpha, int num_links, int nmo);

void build_S_diag(double* S_diag, int num_alpha, int nmo, int N_ac,int n_o_ac,int n_o_in, double shift);
void gram_schmidt_orthogonalization(double* Q, int rows, int cols);
void gram_schmidt_add(double* Q, int rows, int cols, int rows2);
void davidson(double* h1e, double* h2e, double* d_cmo, double* Hdiag, double* eigenvals, double* eigenvecs, int* table, 
		int* table_creation, int* table_annihilation, int *constint, double *constdouble, callback_ build_sigma);

void build_one_rdm(double* eigvec, double* D, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2);
void build_two_rdm(double* eigvec, double* D, int* table, int N_ac, int n_o_ac, int n_o_in, int num_photon, int state_p1, int state_p2);
void build_symmetrized_active_rdm(double* eigvec, double* D_tu, double* D_tuvw, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2); 
void build_active_photon_electron_one_rdm(double* eigvec, double* Dpe_tu, int* table, int N_ac, int n_o_ac, int num_photon, int state_p1, int state_p2); 

void get_roots(double* h1e, double* h2e, double* d_cmo, double* Hdiag, double* eigenvals, double* eigenvecs, int* table, 
		int* table_creation, int* table_annihilation, int *constint, double *constdouble);
void symmetric_eigenvalue_problem(double* A, int N, double* eig);
void getMemory2(unsigned long* currRealMem, unsigned long* peakRealMem, unsigned long* currVirtMem, unsigned long* peakVirtMem);


