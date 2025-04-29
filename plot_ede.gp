set terminal pngcairo enhanced font 'Helvetica,12' size 1200,600
set output 'ede_evolution.png'
set multiplot layout 1,2
set logscale x
set xlabel 'Redshift z'
set ylabel '{/Symbol W}_{ede}'
set title 'Early Dark Energy Density'
plot 'ede_data.txt' using 1:3 with lines lw 2 title '{/Symbol W}_{ede}(z)', \
     0.700000 lt 2 title '{/Symbol W}_{ede,0}', \
     0.020000 lt 3 title '{/Symbol W}_e^{EDE}'
set xlabel 'Redshift z'
set ylabel 'w_{ede}'
set title 'EDE Equation of State'
set yrange [-1.1:0]
plot 'ede_data.txt' using 1:4 with lines lw 2 title 'w_{ede}(z)', \
     -1.000000 lt 2 title 'w_0', \
     -1.0/3.0 lt 3 title '-1/3'
unset multiplot
