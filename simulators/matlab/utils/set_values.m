div_factor = 100;
delay = 50/div_factor;
amplitude = 2;
pulse_width = 100/div_factor;
interphase_gap = 100/div_factor;
decay_width = 50/div_factor;
k = 0.99;
ramp_width = 150/div_factor;

y = zeros(500,1);
t = (1:500)/100;

for i=1:500
    y(i) = pulse_exp_ramp(t(i), delay, amplitude, pulse_width, interphase_gap, decay_width, k, ramp_width);

end
plot(t, y)
hold on
