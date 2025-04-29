set terminal pngcairo enhanced font 'Helvetica,12' size 800,600
set output 'w_ede_evolution.png'
set logscale x
set xlabel 'Redshift z'
set ylabel 'w_{ede}'
set title 'Equation of State for Early Dark Energy'
set yrange [-1.1:0]
set key top left
plot 'ede_data_0.txt' using 1:4 with lines lw 2 title 'Ω_eEDE=0.00', \
     'ede_data_1.txt' using 1:4 with lines lw 2 title 'Ω_eEDE=0.00', \
     'ede_data_2.txt' using 1:4 with lines lw 2 title 'Ω_eEDE=0.00', \
     'ede_data_3.txt' using 1:4 with lines lw 2 title 'Ω_eEDE=0.00', \
     'ede_data_4.txt' using 1:4 with lines lw 2 title 'Ω_eEDE=0.01', \
     'ede_data_5.txt' using 1:4 with lines lw 2 title 'Ω_eEDE=0.01', -1.000000 lt 2 title 'w_0', -1.0/3.0 lt 3 title '-1/3'
