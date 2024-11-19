#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <omp.h>
#include <gsl/gsl_integration.h>

// Constants
const double ALPHA2PI = 7.2973525693e-3 / M_PI;
const double emass = 5.1099895e-4;  // Electron mass in GeV
const double pmass = 0.938272081;   // Proton mass in GeV
const double pi0mass = 0.1349768;   // Pion mass in GeV
const double q2emax = 100000.0;
const double q2pmax = 10.0;
const double MN_max = 10.0;

// ALLM parameters
const double Mass2_0 = 0.31985;
const double Mass2_P = 49.457;
const double Mass2_R = 0.15052;
const double Q2_0 = 0.52544;
const double Lambda2 = 0.06527;

const double Ccp[] = {0.28067, 0.22291, 2.1979};
const double Cap[] = {-0.0808, -0.44812, 1.1709};
const double Cbp[] = {0.36292, 1.8917, 1.8439};
const double Ccr[] = {0.80107, 0.97307, 3.4942};
const double Car[] = {0.58400, 0.37888, 2.6063};
const double Cbr[] = {0.01147, 3.7582, 0.49338};

// Helper functions for ALLM
double tvalue(double Q2) {
    return std::log(std::log((Q2 + Q2_0) / Lambda2) / std::log(Q2_0 / Lambda2));
}

double type1(double tval, const double* tuple) {
    return tuple[0] + tuple[1] * std::pow(tval, tuple[2]);
}

double type2(double tval, const double* tuple) {
    return tuple[0] + (tuple[0] - tuple[1]) * (1.0 / (1.0 + std::pow(tval, tuple[2])) - 1.0);
}

double aP(double tval) { return type2(tval, Cap); }
double bP(double tval) { return type1(tval, Cbp); }
double cP(double tval) { return type2(tval, Ccp); }
double aR(double tval) { return type1(tval, Car); }
double bR(double tval) { return type1(tval, Cbr); }
double cR(double tval) { return type1(tval, Ccr); }

double xP(double xbj, double Q2) {
    if (xbj == 0) return -1.0;
    double xPinv = 1.0 + Q2 / (Q2 + Mass2_P) * (1.0 / xbj - 1.0);
    return 1.0 / xPinv;
}

double xR(double xbj, double Q2) {
    if (xbj == 0) return -1.0;
    double xPinv = 1.0 + Q2 / (Q2 + Mass2_R) * (1.0 / xbj - 1.0);
    return 1.0 / xPinv;
}

double allm_f2P(double xbj, double Q2) {
    double tval = tvalue(Q2);
    return cP(tval) * std::pow(xP(xbj, Q2), aP(tval)) * std::pow(1.0 - xbj, bP(tval));
}

double allm_f2R(double xbj, double Q2) {
    double tval = tvalue(Q2);
    return cR(tval) * std::pow(xR(xbj, Q2), aR(tval)) * std::pow(1.0 - xbj, bR(tval));
}

double allm_f2(double xbj, double Q2) {
    return Q2 / (Q2 + Mass2_0) * (allm_f2P(xbj, Q2) + allm_f2R(xbj, Q2));
}

// Photon virtuality functions
double qmin2_electron(double mass, double y) {
    if (y >= 1) return std::numeric_limits<double>::infinity();
    return mass * mass * y * y / (1.0 - y);
}

double qmin2_proton(double MN, double y) {
    if (y >= 1) return std::numeric_limits<double>::infinity();
    return ((MN * MN) / (1.0 - y) - pmass * pmass) * y;
}

// Compute yp and Jacobian
double compute_yp(double W, double Q2e, double Q2p, double ye, double Ee, double Ep, double MN) {
    double numerator = W * W + Q2e + Q2p - (Q2e * (Q2p + MN * MN - pmass * pmass)) / (4 * Ee * Ep);
    double denominator = ye * 4 * Ee * Ep;
    return numerator / denominator;
}

double compute_jacobian(double ye, double Ee, double Ep, double W) {
    return std::abs(2 * ye * Ee * Ep / W);
}

// Flux from electron
double flux_y_electron(double ye, double lnQ2e) {
    double Q2e = std::exp(lnQ2e);
    if (ye <= 0 || ye >= 1) return 0.0;
    double qmin2v = qmin2_electron(emass, ye);
    if (qmin2v <= 0 || Q2e < qmin2v || Q2e > q2emax) return 0.0;
    double flux = ALPHA2PI / (ye * Q2e) * ((1 - ye) * (1 - qmin2v / Q2e) + 0.5 * ye * ye);
    return flux * Q2e;
}

// Flux from proton
double flux_y_proton(double yp, double lnQ2p, double MN) {
    double Q2p = std::exp(lnQ2p);
    double xbj = Q2p / (MN * MN - pmass * pmass + Q2p);
    if (yp <= 0 || yp >= 1) return 0.0;
    double qmin2p = qmin2_proton(MN, yp);
    if (qmin2p <= 0 || Q2p < qmin2p || Q2p > q2pmax) return 0.0;
    double FE = allm_f2(xbj, Q2p) * (2 * MN / (MN * MN - pmass * pmass + Q2p));
    double FM = allm_f2(xbj, Q2p) * (2 * MN * (MN * MN - pmass * pmass + Q2p)) / (Q2p * Q2p);
    double flux = ALPHA2PI / (yp * Q2p) * ((1 - yp) * (1 - qmin2p / Q2p) * FE + 0.5 * yp * yp * FM);
    return flux * Q2p;
}

// Define structure to hold parameters for GSL integration
struct Params {
    double W, eEbeam, pEbeam, ye, Q2e, MN;
};

// Nested integrands for GSL integration
double lnQ2p_integrand(double lnQ2p, void *params) {
    Params *p = (Params *)params;
    double Q2p = std::exp(lnQ2p);
    double yp_value = compute_yp(p->W, p->Q2e, Q2p, p->ye, p->eEbeam, p->pEbeam, p->MN);
    
    if (yp_value <= 0 || yp_value >= 1) return 0.0;

    double jacobian = compute_jacobian(p->ye, p->eEbeam, p->pEbeam, p->W);
    double proton_flux = flux_y_proton(yp_value, lnQ2p, p->MN);
    
    return proton_flux / jacobian;
}

double MN_integrand(double MN, void *params) {
    Params *p = (Params *)params;
    p->MN = MN;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    double result, error;

    gsl_function F;
    F.function = &lnQ2p_integrand;
    F.params = params;

    double qmin2p = qmin2_proton(MN, 0.01);
    double lnQ2p_min = std::log(qmin2p);
    double lnQ2p_max = std::log(q2pmax);

    gsl_integration_qag(&F, lnQ2p_min, lnQ2p_max, 0, 1e-4, 1000, 6, w, &result, &error);
    gsl_integration_workspace_free(w);

    return result;
}

double lnQ2e_integrand(double lnQ2e, void *params) {
    Params *p = (Params *)params;
    p->Q2e = std::exp(lnQ2e);

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    double result, error;

    gsl_function F;
    F.function = &MN_integrand;
    F.params = params;

    double MN_min = pmass + pi0mass;
    double MN_max = MN_max;

    gsl_integration_qag(&F, MN_min, MN_max, 0, 1e-4, 1000, 6, w, &result, &error);
    gsl_integration_workspace_free(w);

    return result * flux_y_electron(p->ye, lnQ2e);
}

double ye_integrand(double ye, void *params) {
    Params *p = (Params *)params;
    p->ye = ye;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    double result, error;

    gsl_function F;
    F.function = &lnQ2e_integrand;
    F.params = params;

    double qmin2e = qmin2_electron(emass, ye);
    double lnQ2e_min = std::log(qmin2e);
    double lnQ2e_max = std::log(q2emax);

    gsl_integration_qag(&F, lnQ2e_min, lnQ2e_max, 0, 1e-4, 1000, 6, w, &result, &error);
    gsl_integration_workspace_free(w);

    return result;
}

// Main function to compute photon-photon luminosity spectrum
double flux_el_yy_atW(double W, double eEbeam, double pEbeam) {
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    double result, error;

    Params params = {W, eEbeam, pEbeam, 0, 0, 0};

    gsl_function F;
    F.function = &ye_integrand;
    F.params = &params;

    double s_cms = 4.0 * eEbeam * pEbeam;
    double ye_min = W * W / s_cms;
    double ye_max = 1.0;

    gsl_integration_qag(&F, ye_min, ye_max, 0, 1e-4, 1000, 6, w, &result, &error);
    gsl_integration_workspace_free(w);

    return result;
}

int main() {
    double eEbeam = 50.0;
    double pEbeam = 7000.0;

    std::vector<double> W_values;
    for (int i = 0; i <= 100; ++i) {
        W_values.push_back(std::pow(10.0, 1.0 + i * 2.0 / 100.0));
    }

    std::vector<double> luminosity_values(W_values.size());

    #pragma omp parallel for
    for (size_t i = 0; i < W_values.size(); ++i) {
        luminosity_values[i] = flux_el_yy_atW(W_values[i], eEbeam, pEbeam);
    }

    std::ofstream outfile("Jacobian_Krzysztof_Inelastic_Updated.txt");
    outfile << "# W [GeV]    S_yy [GeV^-1]\n";
    for (size_t i = 0; i < W_values.size(); ++i) {
        outfile << W_values[i] << "    " << luminosity_values[i] << "\n";
    }
    outfile.close();

    std::cout << "Calculation completed and saved to Jacobian_Krzysztof_Inelastic_Updated.txt\n";
    return 0;
}
