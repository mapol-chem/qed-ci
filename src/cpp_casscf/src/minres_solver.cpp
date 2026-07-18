#include "casscf/minres_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace casscf {

namespace {
double norm2(double a, double b) { return std::sqrt(a * a + b * b); }
} // namespace

MinresResult minres_solve(const Matrix& A, const Vector& b, double rtol, int maxiter) {
    const int n = static_cast<int>(A.rows());
    if (maxiter < 0) {
        maxiter = 5 * n; // scipy's default when maxiter isn't passed
    }

    const double eps = std::numeric_limits<double>::epsilon();

    MinresResult result;
    Vector x = Vector::Zero(n);

    // x0 = None path: r1 = b.copy(); y = psolve(r1) = r1 (no preconditioner).
    Vector r1 = b;
    Vector y = r1;

    double beta1 = r1.dot(y);
    if (beta1 < 0.0) {
        // scipy: "indefinite preconditioner" -- mathematically unreachable
        // with M = I, kept only for structural fidelity.
        result.x = x;
        result.istop = -2;
        return result;
    }
    if (beta1 == 0.0) {
        result.x = x;
        result.istop = 0;
        return result;
    }
    if (b.norm() == 0.0) {
        result.x = b; // == 0
        result.istop = 0;
        return result;
    }

    beta1 = std::sqrt(beta1);

    double oldb = 0.0;
    double beta = beta1;
    double dbar = 0.0;
    double epsln = 0.0;
    double phibar = beta1;
    double rhs1 = beta1;
    double rhs2 = 0.0;
    double tnorm2 = 0.0;
    double gmax = 0.0;
    double gmin = std::numeric_limits<double>::max();
    double cs = -1.0;
    double sn = 0.0;
    Vector w = Vector::Zero(n);
    Vector w2 = Vector::Zero(n);
    Vector r2 = r1;

    int istop = 0;
    int itn = 0;

    while (itn < maxiter) {
        ++itn;

        const double s = 1.0 / beta;
        Vector v = s * y;

        y = A * v;
        if (itn >= 2) {
            y = y - (beta / oldb) * r1;
        }

        const double alfa = v.dot(y);
        y = y - (alfa / beta) * r2;
        r1 = r2;
        r2 = y;
        y = r2; // psolve(r2), no preconditioner
        oldb = beta;
        double beta_sq = r2.dot(y);
        // scipy raises "non-symmetric matrix" if beta_sq < 0; for a
        // symmetric A that's only reachable via roundoff on a near-singular
        // system -- clamp instead of raising, see header doc comment.
        beta = std::sqrt(std::max(beta_sq, 0.0));
        tnorm2 += alfa * alfa + oldb * oldb + beta * beta;

        if (itn == 1 && beta / beta1 <= 10.0 * eps) {
            istop = -1;
        }

        const double oldeps = epsln;
        const double delta = cs * dbar + sn * alfa;
        const double gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;
        const double root = norm2(gbar, dbar);

        double gamma = norm2(gbar, beta);
        gamma = std::max(gamma, eps);
        cs = gbar / gamma;
        sn = beta / gamma;
        const double phi = cs * phibar;
        phibar = sn * phibar;

        const double denom = 1.0 / gamma;
        Vector w1 = w2;
        w2 = w;
        w = (v - oldeps * w1 - delta * w2) * denom;
        x = x + phi * w;

        gmax = std::max(gmax, gamma);
        gmin = std::min(gmin, gamma);
        const double z = rhs1 / gamma;
        rhs1 = rhs2 - delta * z;
        rhs2 = -epsln * z;

        const double Anorm = std::sqrt(tnorm2);
        const double ynorm = x.norm();
        const double epsx = Anorm * ynorm * eps;

        const double qrnorm = phibar;
        const double rnorm = qrnorm;
        const double test1 = (ynorm == 0.0 || Anorm == 0.0)
                                  ? std::numeric_limits<double>::infinity()
                                  : rnorm / (Anorm * ynorm);
        const double test2 =
            (Anorm == 0.0) ? std::numeric_limits<double>::infinity() : root / Anorm;

        const double Acond = gmax / gmin;

        if (istop == 0) {
            const double t1 = 1.0 + test1;
            const double t2 = 1.0 + test2;
            if (t2 <= 1.0) istop = 2;
            if (t1 <= 1.0) istop = 1;

            if (itn >= maxiter) istop = 6;
            if (Acond >= 0.1 / eps) istop = 4;
            if (epsx >= beta1) istop = 3;
            if (test2 <= rtol) istop = 2;
            if (test1 <= rtol) istop = 1;
        }

        if (istop != 0) break;
    }

    result.x = x;
    result.istop = istop;
    result.iterations = itn;
    return result;
}

} // namespace casscf
