This error means that your code is declaring dsyev_ with a different signature than the one provided by Apple's Accelerate framework. Specifically:

Your declaration:

cpp
Copy
Edit
void dsyev_(char* jobz, char* uplo, int* n, double* A, int* lda, double* eig, double* work, int* lwork, int* info);
Uses int* for n, lda, lwork, and info.
Apple's dsyev_ declaration in lapack.h:

c
Copy
Edit
extern void dsyev_(char *jobz, char *uplo, __CLPK_integer *n, __CLPK_doublereal *a, __CLPK_integer *lda, __CLPK_doublereal *w, __CLPK_doublereal *work, __CLPK_integer *lwork, __CLPK_integer *info);
Uses __CLPK_integer instead of int.
Uses __CLPK_doublereal instead of double.
Solution
Instead of declaring dsyev_ yourself, include the correct header:

cpp
Copy
Edit
#include <Accelerate/Accelerate.h>
Then, use __CLPK_integer and __CLPK_doublereal for compatibility:

cpp
Copy
Edit
__CLPK_integer n = ...;
__CLPK_integer lda = ...;
__CLPK_integer lwork = ...;
__CLPK_integer info;
__CLPK_doublereal* A = ...;
__CLPK_doublereal* eig = ...;
__CLPK_doublereal* work = ...;

dsyev_("V", "U", &n, A, &lda, eig, work, &lwork, &info);
This ensures compatibility with the LAPACK implementation in Apple's Accelerate framework.``
