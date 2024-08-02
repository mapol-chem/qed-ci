#include <stdint.h>
#include <stdbool.h>

void full_transformation_macroiteration(double* U, double* h2e, double* J, double *K, int* index_map_pq, int* index_map_kl, int nmo, int n_occupied);
void full_transformation_internal_optimization(double* U, double* J, double *K, double* h, double* d_cmo, 
		double* J1, double* K1, double* h1, double* d_cmo1, int* index_map_ab, int* index_map_kl, int nmo, int n_occupied);
void build_sigma_reduced(double* U, double* A_tilde, int* index_map, double* G, double* R_reduced, double* sigma_reduced, int num_states, int pointer, int nmo, int index_map_size, int n_occupied);
