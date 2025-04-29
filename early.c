#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Cosmological parameters
const double Omega_ede0 = 0.7;
const double Omega_eEDE_values[] = {0.001, 0.002, 0.003, 0.004, 0.005, 0.006};
const int num_Omega_eEDE = sizeof(Omega_eEDE_values)/sizeof(Omega_eEDE_values[0]);
const double Omega_m0 = 0.3;
const double w0 = -1.0;
const double a_eq = 1.0/3400.0;
const double eps = 1e-9;

// Function to calculate Omega_ede(a)
double Omega_ede(double a, double Omega_eEDE) {
    double term1 = (Omega_ede0 - (Omega_eEDE * (1.0 - pow(a, -3.0*w0))));
    double term2 = (Omega_ede0 + (Omega_m0 * pow(a, 3.0*w0)));
    double term3 = Omega_eEDE * (1.0 - pow(a, -3.0*w0));
    return (term1/term2) + term3;
}

// Numerical derivative of ln(Omega_ede) with respect to ln(a)
double dlnOmega_dlna(double a, double Omega_eEDE, double eps) {
    double log_Omega = log(Omega_ede(a, Omega_eEDE));
    double log_Omega_plus = log(Omega_ede(a * (1.0 + eps), Omega_eEDE));
    return (log_Omega_plus - log_Omega) / log(1.0 + eps);
}

// Function to calculate w_ede(a)
double w_ede(double a, double Omega_eEDE) {
    double omega = Omega_ede(a, Omega_eEDE);
    double deriv = dlnOmega_dlna(a, Omega_eEDE, eps);
    double term1 = -deriv/(3.0*(1.0 - omega));
    double term2 = a_eq/(3.0*(a + a_eq));
    return term1 + term2;
}

int main() {
    const int n_points = 1000;
    double z[n_points];
    double a[n_points];
    
    // Create redshift range (logarithmic)
    double loga_min = log10(pow(10,-6));
    double loga_max = log10(0);
    double delta_loga = (loga_max - loga_min)/(n_points-1);
    
    for (int i = 0; i < n_points; i++) {
        a[i] = pow(10, loga_max - i*delta_loga);
        z[i] = (1.0/(a[i])) - 1;
    }
    
    // Create data files for each Omega_eEDE value
    for (int j = 0; j < num_Omega_eEDE; j++) {
        char filename[50];
        sprintf(filename, "ede_data_%d.txt", j);
        
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {
            printf("Error opening file %s!\n", filename);
            continue;
        }
        
        fprintf(fp, "# z\t a\t Omega_ede\t w_ede\n");
        for (int i = 0; i < n_points; i++) {
            double Omega_ede_val = Omega_ede(a[i], Omega_eEDE_values[j]);
            double w_ede_val = w_ede(a[i], Omega_eEDE_values[j]);
            fprintf(fp, "%e\t %e\t %e\t %e\n", z[i], a[i], Omega_ede_val, w_ede_val);
        }
        fclose(fp);
    }
    
    // Create Gnuplot script (ONLY for w_ede)
    FILE *gp = fopen("plot_ede.gp", "w");
    if (gp == NULL) {
        printf("Error creating gnuplot script!\n");
        return 1;
    }
    
    fprintf(gp, "set terminal pngcairo enhanced font 'Helvetica,12' size 800,600\n");
    fprintf(gp, "set output 'w_ede_evolution.png'\n");
    fprintf(gp, "set logscale x\n");
    fprintf(gp, "set xlabel 'Redshift z'\n");
    fprintf(gp, "set ylabel 'w_{ede}'\n");
    fprintf(gp, "set title 'Equation of State for Early Dark Energy'\n");
    fprintf(gp, "set yrange [-1.1:0]\n");
    fprintf(gp, "set key top left\n");  // Adjust legend position
    
    // Plot command for w_ede
    fprintf(gp, "plot ");
    for (int j = 0; j < num_Omega_eEDE; j++) {
        fprintf(gp, "'ede_data_%d.txt' using 1:4 with lines lw 2 title 'Ω_eEDE=%.2f'", j, Omega_eEDE_values[j]);
        if (j < num_Omega_eEDE-1) fprintf(gp, ", \\\n     ");
    }
    fprintf(gp, ", %f lt 2 title 'w_0', -1.0/3.0 lt 3 title '-1/3'\n", w0);
    
    fclose(gp);
    
    // Execute Gnuplot
    system("gnuplot plot_ede.gp");
    
    printf("Plot generated as w_ede_evolution.png\n");
    
    return 0;
}
